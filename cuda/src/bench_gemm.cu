// ============================================================================
// bench_gemm.cu  --  GEMM/GEMV baseline profiler (Track A, phase A1.3)
//
// Measures the current naive FP16 dot-product path (device_matvec /
// device_matvec_batched from utils.h) against tensor-core cuBLAS (cublasGemmEx
// with CUBLAS_GEMM_DEFAULT_TENSOR_OP) at the real projection shapes of a
// transformer layer, for the batch sizes that occur in practice:
//
//   M = 1    single-token decode
//   M = 5    speculative verify (spec_k = 4  ->  k + 1)
//   M = 128  prefill chunk
//
// The point is to answer "which ops matter, and how much is left on the table"
// before porting anything to cuBLASLt/CUTLASS.  It is a standalone executable;
// it does not touch the inference paths.
//
// Build (from cuda/):
//   nvcc -std=c++17 -O3 --use_fast_math -Iinclude -arch=sm_86 \
//        -o bench_gemm.exe src/bench_gemm.cu -lcublas
// ============================================================================

#include "utils.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ---- Naive-path launch wrappers (single thread block, mirroring real usage) --

__global__ void matvec_naive_kernel(const float* x, const half* W,
                                    float* out, int d_in, int d_out) {
    device_matvec(x, W, out, d_in, d_out);
}

__global__ void matvec_batched_kernel(const float* gx, const half* W,
                                      float* gout, int d_in, int d_out, int B) {
    device_matvec_batched(gx, W, gout, d_in, d_out, B);
}

// Sequential per-token prefill cost: the current engine runs one single-token
// forward per prompt token, so M tokens = M back-to-back device_matvec calls.
__global__ void matvec_prefill_kernel(const float* gx, const half* W,
                                      float* gout, int d_in, int d_out, int M) {
    for (int m = 0; m < M; m++)
        device_matvec(gx + (size_t)m * d_in, W, gout + (size_t)m * d_out,
                      d_in, d_out);
}

// ----------------------------------------------------------------------------

struct Op { const char* name; int d_in; int d_out; };

static float time_naive(int M, const float* d_x, const half* d_W, float* d_obuf,
                        int d_in, int d_out, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    auto launch = [&]() {
        if (M == 1)
            matvec_naive_kernel<<<1, BLOCK_THREADS>>>(d_x, d_W, d_obuf, d_in, d_out);
        else if (M <= MAX_VERIFY_BATCH)
            matvec_batched_kernel<<<1, BLOCK_THREADS>>>(d_x, d_W, d_obuf, d_in, d_out, M);
        else
            matvec_prefill_kernel<<<1, BLOCK_THREADS>>>(d_x, d_W, d_obuf, d_in, d_out, M);
    };

    for (int i = 0; i < 2; i++) launch();          // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

// C[M,N] (row-major, fp32) = X[M,K] (fp16) @ W[K,N] (fp16), tensor-core path.
// Column-major cuBLAS computes C^T = W^T * X^T via one OP_N/OP_N call.
static float time_cublas(cublasHandle_t h, int M, const half* d_X, const half* d_W,
                         float* d_C, int K, int N, int iters) {
    const float alpha = 1.f, beta = 0.f;
    auto launch = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_W, CUDA_R_16F, N,
            d_X, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    for (int i = 0; i < 5; i++) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: stream results live
    // Llama 3.2 1B shapes (override d_model/vocab via argv if desired).
    int d_model = 2048;
    int d_ff    = 8192;
    int head_dim_ = 64;
    int n_heads = 32, n_kv_heads = 8;
    int vocab   = 128256;
    int n_layers = 16;

    const int iters = 50;
    const int Ms[]  = {1, 5, 128};

    printf("GEMM baseline: naive device_matvec vs cuBLAS tensor-op (sm_86)\n");
    printf("d_model=%d d_ff=%d vocab=%d n_layers=%d  (kv_dim=%d)\n\n",
           d_model, d_ff, vocab, n_layers, n_kv_heads * head_dim_);

    const Op ops[] = {
        {"Wq            ", d_model, n_heads * head_dim_},
        {"Wk/Wv expanded", d_model, d_model},
        {"Wk/Wv native  ", d_model, n_kv_heads * head_dim_},
        {"Wo            ", d_model, d_model},
        {"W_gate        ", d_model, d_ff},
        {"W_up          ", d_model, d_ff},
        {"W_down        ", d_ff,    d_model},
        {"output_proj   ", d_model, vocab},
    };
    const int n_ops = sizeof(ops) / sizeof(ops[0]);

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    // Largest buffers sized for the biggest op / batch up front.
    int max_din = d_ff, max_dout = vocab, max_M = 128;
    std::vector<half>  h_W((size_t)max_din * max_dout);
    std::vector<half>  h_Xh((size_t)max_M * max_din);
    std::vector<float> h_Xf((size_t)max_M * max_din);
    for (auto& w : h_W)  w  = __float2half(dist(rng));
    for (size_t i = 0; i < h_Xf.size(); i++) { float v = dist(rng); h_Xf[i] = v; h_Xh[i] = __float2half(v); }

    half  *d_W, *d_Xh;
    float *d_Xf, *d_out;
    CUDA_CHECK(cudaMalloc(&d_W,   (size_t)max_din * max_dout * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Xh,  (size_t)max_M   * max_din  * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Xf,  (size_t)max_M   * max_din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)max_M   * max_dout * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W,  h_W.data(),  h_W.size()  * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Xh, h_Xh.data(), h_Xh.size() * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Xf, h_Xf.data(), h_Xf.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int mi = 0; mi < 3; mi++) {
        int M = Ms[mi];
        const char* tag = (M == 1) ? "decode" : (M == 5 ? "verify" : "prefill");
        printf("== M=%-3d (%s) ==\n", M, tag);
        printf("  %-15s %10s %12s %12s %9s   %s\n",
               "op", "shape", "naive(us)", "cublas(us)", "speedup", "cublas TFLOP/s");
        int naive_iters = (M >= 64) ? 3 : iters;   // naive M=128 path is very slow
        for (int i = 0; i < n_ops; i++) {
            const Op& o = ops[i];
            float nus = time_naive(M, d_Xf, d_W, d_out, o.d_in, o.d_out, naive_iters) * 1e3f;
            float cus = time_cublas(h, M, d_Xh, d_W, d_out, o.d_in, o.d_out, iters) * 1e3f;
            double flop = 2.0 * M * o.d_in * o.d_out;
            double tflops = flop / (cus * 1e-6) / 1e12;
            char shape[32];
            snprintf(shape, sizeof(shape), "%dx%d", o.d_in, o.d_out);
            printf("  %-15s %10s %12.1f %12.1f %8.1fx   %8.2f\n",
                   o.name, shape, nus, cus, nus / cus, tflops);
        }
        printf("\n");
    }

    // Per-token decode budget (M=1): where does the time actually go?
    printf("== per-token decode budget (M=1, naive) ==\n");
    double t_attn = 0, t_mlp = 0, t_out = 0;
    for (int i = 0; i < n_ops; i++) {
        const Op& o = ops[i];
        if (std::string(o.name).find("expanded") != std::string::npos) continue; // count native kv only
        float nus = time_naive(1, d_Xf, d_W, d_out, o.d_in, o.d_out, iters) * 1e3f;
        std::string nm(o.name);
        if (nm.find("output") != std::string::npos)      t_out  += nus;
        else if (nm.find("W_") != std::string::npos)      t_mlp  += nus * n_layers;
        else                                              t_attn += nus * n_layers;
    }
    double total = t_attn + t_mlp + t_out;
    printf("  attention proj (Wq/Wk/Wv/Wo) x%d layers : %8.1f us  (%4.1f%%)\n", n_layers, t_attn, 100*t_attn/total);
    printf("  MLP proj (gate/up/down)       x%d layers : %8.1f us  (%4.1f%%)\n", n_layers, t_mlp, 100*t_mlp/total);
    printf("  output_proj                             : %8.1f us  (%4.1f%%)\n", t_out, 100*t_out/total);
    printf("  ---------------------------------------------------------\n");
    printf("  total GEMV                              : %8.1f us\n", total);

    cudaFree(d_W); cudaFree(d_Xh); cudaFree(d_Xf); cudaFree(d_out);
    cublasDestroy(h);
    return 0;
}
