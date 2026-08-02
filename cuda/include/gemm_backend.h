 #pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "utils.h"

// GemmBackend — thin wrapper over cublasGemmEx for tensor-op GEMM.
//
// All GEMMs compute  C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
// where A is row-major FP16, B is row-major FP16, C is row-major FP32.
//
// cuBLAS is column-major, so we compute C^T = B^T * A^T with OP_N/OP_N:
//   A_cublas = B  (col-major N×K, ld=N)
//   B_cublas = A  (col-major K×M, ld=K)
//   C_cublas = C  (col-major N×M, ld=N)
//
// This matches the weight layout (row-major half*) and the hidden state
// layout (row-major float*) used throughout the engine.

struct GemmBackend {
    cublasHandle_t handle;
    cudaStream_t   stream;

    void init(cublasHandle_t h, cudaStream_t s) {
        handle = h;
        stream = s;
    }
};

// C[M,N] (fp32) = X[M,K] (fp16) @ W[K,N] (fp16)
// X: row-major fp16, stride K.  W: row-major fp16, stride N.  C: row-major fp32, stride N.
inline void gemm_backend_gemm(const GemmBackend& gb,
                              int M, int N, int K,
                              const half* X, const half* W, float* C) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(gb.handle, gb.stream));
    CUBLAS_CHECK(cublasGemmEx(
        gb.handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16F, N,       // B^T in col-major
        X, CUDA_R_16F, K,       // A^T in col-major
        &beta,
        C, CUDA_R_32F, N,       // C^T in col-major
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Single-token GEMV: out[N] (fp32) = x[K] (fp16) @ W[K,N] (fp16)
// Equivalent to gemm with M=1.
inline void gemm_backend_gemv(const GemmBackend& gb,
                              int N, int K,
                              const half* x, const half* W, float* out) {
    gemm_backend_gemm(gb, 1, N, K, x, W, out);
}

// Strided batched: B independent GEMMs, each C_b[M,N] = X_b[M,K] @ W[K,N].
// X has stride strideX between batches, C has stride strideC.
// W is the same for all batches (broadcast).
inline void gemm_backend_gemm_strided(const GemmBackend& gb,
                                      int M, int N, int K, int B,
                                      const half* X, long long strideX,
                                      const half* W,
                                      float* C, long long strideC) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(gb.handle, gb.stream));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        gb.handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16F, N, 0,          // W is broadcast (stride 0)
        X, CUDA_R_16F, K, strideX,
        &beta,
        C, CUDA_R_32F, N, strideC,
        B,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
