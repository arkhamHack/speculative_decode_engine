#pragma once
#include "config.h"
#include <cuda_fp16.h>
#include <cfloat>

// ============================================================================
// Warp-level reductions using __shfl_xor_sync
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================================
// Block-level reductions  (all threads contribute one value)
// Result is broadcast to all threads via smem[0].
// smem must have at least ceil(blockDim.x / WARP_SIZE) float slots.
// ============================================================================

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int tid     = threadIdx.x;
    int lane    = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    int n_warps = blockDim.x / WARP_SIZE;
    val = (tid < n_warps) ? smem[tid] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (tid == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int tid     = threadIdx.x;
    int lane    = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    int n_warps = blockDim.x / WARP_SIZE;
    val = (tid < n_warps) ? smem[tid] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);

    if (tid == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ============================================================================
// RMSNorm:  out[i] = (x[i] / rms(x)) * weight[i]
//
// Handles d > blockDim.x via stride loop: each thread accumulates
// its assigned elements into the shared sum-of-squares, then normalises
// the same elements in a second pass.
// ============================================================================

__device__ __forceinline__
void device_rmsnorm(const float* x, const half* weight, float* out,
                    int d, float* smem) {
    int tid = threadIdx.x;

    // Accumulate squared elements (stride loop for d > BLOCK_THREADS)
    float ss = 0.0f;
    for (int i = tid; i < d; i += blockDim.x)
        ss += x[i] * x[i];
    ss = block_reduce_sum(ss, smem);       // result in all threads

    float rms = rsqrtf(ss / (float)d + 1e-6f);

    for (int i = tid; i < d; i += blockDim.x)
        out[i] = x[i] * rms * __half2float(weight[i]);
    __syncthreads();
}

// ============================================================================
// In-place softmax over data[0..n-1] in shared memory.
// Handles n > blockDim.x via stride loop.
// ============================================================================

__device__ __forceinline__
void block_softmax_inplace(float* data, int n, float* smem) {
    int tid = threadIdx.x;

    // --- Pass 1: find max (for numerical stability) ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x)
        local_max = fmaxf(local_max, data[i]);
    float max_val = block_reduce_max(local_max, smem);

    // --- Pass 2: exp and local partial sum ---
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(data[i] - max_val);
        data[i]   = e;
        local_sum += e;
    }
    float sum_val = block_reduce_sum(local_sum, smem);

    // --- Pass 3: normalize ---
    for (int i = tid; i < n; i += blockDim.x)
        data[i] /= sum_val;
    __syncthreads();
}

// ============================================================================
// Argmax over a global-memory array data[0..n-1].
// Handles n >> blockDim.x via stride loop.
//
// scratch must point to 2 * blockDim.x float-worth of shared memory.
// The caller can reuse any scratch portion of the shared memory layout
// (e.g. the scores[] or normed[] region) since they are idle at argmax time.
//
// Technique: store (max_val, best_idx) per thread; binary-tree reduce.
// The int indices are bit-cast into the float scratch array to avoid a
// separate smem allocation (safe in device code with __shared__).
// ============================================================================

__device__ __forceinline__
int global_argmax(const float* data, int n, float* scratch) {
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    float* s_val = scratch;                   // [nthreads] floats for max values
    int*   s_idx = (int*)(scratch + nthreads);// [nthreads] ints  for best indices

    // Each thread scans its stripe
    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < n; i += nthreads) {
        float v = data[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Binary-tree reduction over all threads
    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    return s_idx[0];
}

// ============================================================================
// Matrix-vector multiply:  out[d_out] = x[d_in] @ W[d_in, d_out]
// W stored row-major in half precision: W[row * d_out + col].
// x is in shared memory (float); out may be shared or global memory.
// Each thread computes ceil(d_out / blockDim.x) output elements.
// ============================================================================

__device__ __forceinline__
void device_matvec(const float* x, const half* W,
                   float* out, int d_in, int d_out) {
    int tid = threadIdx.x;
    for (int col = tid; col < d_out; col += blockDim.x) {
        float acc = 0.0f;
        for (int row = 0; row < d_in; row++)
            acc += x[row] * __half2float(W[row * d_out + col]);
        out[col] = acc;
    }
    __syncthreads();
}

// Column slice: out[jc] = x @ W[:, col0+jc], jc in [0, ncol).
// ncol may be smaller than blockDim.x; out must hold ncol floats contiguously.
__device__ __forceinline__
void device_matvec_cols(const float* x, const half* W,
                        int d_in, int d_out,
                        int col0, int ncol,
                        float* out) {
    int tid = threadIdx.x;
    for (int jc = tid; jc < ncol; jc += blockDim.x) {
        float acc = 0.0f;
        int col   = col0 + jc;
        for (int row = 0; row < d_in; row++)
            acc += x[row] * __half2float(W[row * d_out + col]);
        out[jc] = acc;
    }
    __syncthreads();
}

// ============================================================================
// Batched matrix-vector multiply: reads W ONCE for all B input vectors.
// g_x: [B * d_in] contiguous in global memory (batch-major).
// g_out: [B * d_out] contiguous in global memory.
//
// Templatized on compile-time B_CT so the accumulator array is a fixed-size
// register array that the compiler can fully unroll and keep in registers,
// avoiding local-memory (L1/DRAM) spilling that occurs with runtime-sized arrays.
// Runtime dispatch selects the right instantiation for B = 1..MAX_VERIFY_BATCH.
// ============================================================================

template <int B_CT>
__device__ __forceinline__
void device_matvec_batched_T(const float* g_x, const half* W,
                              float* g_out, int d_in, int d_out) {
    int tid = threadIdx.x;
    for (int col = tid; col < d_out; col += blockDim.x) {
        float acc[B_CT];
        #pragma unroll
        for (int b = 0; b < B_CT; b++) acc[b] = 0.f;
        for (int row = 0; row < d_in; row++) {
            float w = __half2float(W[row * d_out + col]);
            #pragma unroll
            for (int b = 0; b < B_CT; b++)
                acc[b] += g_x[b * d_in + row] * w;
        }
        #pragma unroll
        for (int b = 0; b < B_CT; b++)
            g_out[b * d_out + col] = acc[b];
    }
    __syncthreads();
}

__device__ __forceinline__
void device_matvec_batched(const float* g_x, const half* W,
                           float* g_out, int d_in, int d_out, int B) {
    switch (B) {
        case 1:  device_matvec_batched_T<1>(g_x, W, g_out, d_in, d_out); return;
        case 2:  device_matvec_batched_T<2>(g_x, W, g_out, d_in, d_out); return;
        case 3:  device_matvec_batched_T<3>(g_x, W, g_out, d_in, d_out); return;
        case 4:  device_matvec_batched_T<4>(g_x, W, g_out, d_in, d_out); return;
        case 5:  device_matvec_batched_T<5>(g_x, W, g_out, d_in, d_out); return;
        case 6:  device_matvec_batched_T<6>(g_x, W, g_out, d_in, d_out); return;
        case 7:  device_matvec_batched_T<7>(g_x, W, g_out, d_in, d_out); return;
        case 8:  device_matvec_batched_T<8>(g_x, W, g_out, d_in, d_out); return;
        case 9:  device_matvec_batched_T<9>(g_x, W, g_out, d_in, d_out); return;
        default: device_matvec_batched_T<1>(g_x, W, g_out, d_in, d_out); return;
    }
}

// Batched column-slice matvec: out[b][jc] = x[b] @ W[:, col0+jc] for all B inputs.
// g_out: [B * ncol] contiguous.

template <int B_CT>
__device__ __forceinline__
void device_matvec_cols_batched_T(const float* g_x, const half* W,
                                   int d_in, int d_out,
                                   int col0, int ncol,
                                   float* g_out) {
    int tid = threadIdx.x;
    for (int jc = tid; jc < ncol; jc += blockDim.x) {
        float acc[B_CT];
        #pragma unroll
        for (int b = 0; b < B_CT; b++) acc[b] = 0.f;
        int col = col0 + jc;
        for (int row = 0; row < d_in; row++) {
            float w = __half2float(W[row * d_out + col]);
            #pragma unroll
            for (int b = 0; b < B_CT; b++)
                acc[b] += g_x[b * d_in + row] * w;
        }
        #pragma unroll
        for (int b = 0; b < B_CT; b++)
            g_out[b * ncol + jc] = acc[b];
    }
    __syncthreads();
}

__device__ __forceinline__
void device_matvec_cols_batched(const float* g_x, const half* W,
                                int d_in, int d_out,
                                int col0, int ncol,
                                float* g_out, int B) {
    switch (B) {
        case 1:  device_matvec_cols_batched_T<1>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 2:  device_matvec_cols_batched_T<2>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 3:  device_matvec_cols_batched_T<3>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 4:  device_matvec_cols_batched_T<4>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 5:  device_matvec_cols_batched_T<5>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 6:  device_matvec_cols_batched_T<6>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 7:  device_matvec_cols_batched_T<7>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 8:  device_matvec_cols_batched_T<8>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        case 9:  device_matvec_cols_batched_T<9>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
        default: device_matvec_cols_batched_T<1>(g_x, W, d_in, d_out, col0, ncol, g_out); return;
    }
}

// Batched down-projection with accumulation (tiled MLP).
// g_act: [B * ncol] activation slice (SwiGLU output).
// g_accum: [B * d] accumulator (caller must zero before first tile).

template <int B_CT>
__device__ __forceinline__
void device_down_proj_accum_batched_T(const float* g_act, const half* W_down,
                                       int d, int dff, int r0, int ncol,
                                       float* g_accum) {
    int tid = threadIdx.x;
    for (int oc = tid; oc < d; oc += blockDim.x) {
        float dots[B_CT];
        #pragma unroll
        for (int b = 0; b < B_CT; b++) dots[b] = 0.f;
        for (int jc = 0; jc < ncol; jc++) {
            float w = __half2float(W_down[(r0 + jc) * d + oc]);
            #pragma unroll
            for (int b = 0; b < B_CT; b++)
                dots[b] += g_act[b * ncol + jc] * w;
        }
        #pragma unroll
        for (int b = 0; b < B_CT; b++)
            g_accum[b * d + oc] += dots[b];
    }
    __syncthreads();
}

__device__ __forceinline__
void device_down_proj_accum_batched(const float* g_act, const half* W_down,
                                    int d, int dff, int r0, int ncol,
                                    float* g_accum, int B) {
    switch (B) {
        case 1:  device_down_proj_accum_batched_T<1>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 2:  device_down_proj_accum_batched_T<2>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 3:  device_down_proj_accum_batched_T<3>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 4:  device_down_proj_accum_batched_T<4>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 5:  device_down_proj_accum_batched_T<5>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 6:  device_down_proj_accum_batched_T<6>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 7:  device_down_proj_accum_batched_T<7>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 8:  device_down_proj_accum_batched_T<8>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        case 9:  device_down_proj_accum_batched_T<9>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
        default: device_down_proj_accum_batched_T<1>(g_act, W_down, d, dff, r0, ncol, g_accum); return;
    }
}

// ============================================================================
// Scalar half->float element load helper (vectorised load left as TODO)
// ============================================================================

__device__ __forceinline__
void load_half_to_float(const half* src, float* dst, int n, int tid, int stride) {
    for (int i = tid; i < n; i += stride)
        dst[i] = __half2float(src[i]);
}

// ============================================================================
// CUDA error-checking macro
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================================
// Host-side alignment helper
// ============================================================================

inline size_t align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}
