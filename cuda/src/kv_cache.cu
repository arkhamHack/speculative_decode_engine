#include "kv_cache.h"
#include "utils.h"
#include <cstdio>
#include <cstring>

// ============================================================================
// Host API
// ============================================================================

void kv_cache_alloc(KVCache& cache, int n_layers, int d_head,
                    int max_blocks_per_layer) {
    cache.n_layers = n_layers;
    cache.d_head   = d_head;
    cache.next_free_block = 0;

    // Total physical blocks: n_layers * max_blocks_per_layer
    cache.pool_size = n_layers * max_blocks_per_layer;
    size_t elems_per_block = 2 * KV_BLOCK_SIZE * d_head;  // K + V
    size_t pool_bytes = (size_t)cache.pool_size * elems_per_block * sizeof(half);
    CUDA_CHECK(cudaMalloc(&cache.pool, pool_bytes));
    CUDA_CHECK(cudaMemset(cache.pool, 0, pool_bytes));

    // Layer metadata
    CUDA_CHECK(cudaMalloc(&cache.layers, n_layers * sizeof(KVCacheLayer)));
    CUDA_CHECK(cudaMemset(cache.layers, 0, n_layers * sizeof(KVCacheLayer)));

    // Sequence length counter
    CUDA_CHECK(cudaMalloc(&cache.seq_len, sizeof(int)));
    CUDA_CHECK(cudaMemset(cache.seq_len, 0, sizeof(int)));

    // Pre-assign blocks to layers (simple static allocation)
    KVCacheLayer* h_layers = new KVCacheLayer[n_layers];
    memset(h_layers, 0, n_layers * sizeof(KVCacheLayer));
    for (int l = 0; l < n_layers; l++) {
        for (int b = 0; b < max_blocks_per_layer; b++) {
            h_layers[l].block_table[b] = cache.next_free_block++;
        }
        h_layers[l].n_blocks_used = max_blocks_per_layer;
    }
    CUDA_CHECK(cudaMemcpy(cache.layers, h_layers,
                           n_layers * sizeof(KVCacheLayer),
                           cudaMemcpyHostToDevice));
    delete[] h_layers;
}

void kv_cache_free(KVCache& cache) {
    if (cache.pool)   { cudaFree(cache.pool);   cache.pool   = nullptr; }
    if (cache.layers)  { cudaFree(cache.layers);  cache.layers  = nullptr; }
    if (cache.seq_len) { cudaFree(cache.seq_len); cache.seq_len = nullptr; }
}

void kv_cache_reset(KVCache& cache) {
    CUDA_CHECK(cudaMemset(cache.seq_len, 0, sizeof(int)));
}

// ============================================================================
// Device API
// ============================================================================

// Helper: pointer to the start of a physical block's data.
// Layout: [K: KV_BLOCK_SIZE * d_head halfs][V: KV_BLOCK_SIZE * d_head halfs]
__device__ __forceinline__
half* block_ptr(KVCache& cache, int phys_block) {
    size_t elems_per_block = 2 * KV_BLOCK_SIZE * cache.d_head;
    return cache.pool + (size_t)phys_block * elems_per_block;
}

__device__ __forceinline__
const half* block_ptr_const(const KVCache& cache, int phys_block) {
    size_t elems_per_block = 2 * KV_BLOCK_SIZE * cache.d_head;
    return cache.pool + (size_t)phys_block * elems_per_block;
}

__device__ void kv_cache_append(KVCache& cache, int layer,
                                const float* k_vec, const float* v_vec,
                                int current_seq_len) {
    int tid = threadIdx.x;
    int d   = cache.d_head;

    // Logical block index and slot within the block
    int log_block = current_seq_len / KV_BLOCK_SIZE;
    int slot      = current_seq_len % KV_BLOCK_SIZE;

    int phys_block = cache.layers[layer].block_table[log_block];
    half* base = block_ptr(cache, phys_block);

    // K region starts at offset 0; V region starts at KV_BLOCK_SIZE * d
    half* k_dst = base + slot * d;
    half* v_dst = base + KV_BLOCK_SIZE * d + slot * d;

    // Stride loop so d > BLOCK_THREADS is handled correctly.
    for (int i = tid; i < d; i += blockDim.x) {
        k_dst[i] = __float2half(k_vec[i]);
        v_dst[i] = __float2half(v_vec[i]);
    }
    __syncthreads();
}

__device__ void kv_cache_read(const KVCache& cache, int layer, int pos,
                              float* k_out, float* v_out) {
    int tid = threadIdx.x;
    int d   = cache.d_head;

    int log_block = pos / KV_BLOCK_SIZE;
    int slot      = pos % KV_BLOCK_SIZE;

    int phys_block = cache.layers[layer].block_table[log_block];
    const half* base = block_ptr_const(cache, phys_block);

    const half* k_src = base + slot * d;
    const half* v_src = base + KV_BLOCK_SIZE * d + slot * d;

    for (int i = tid; i < d; i += blockDim.x) {
        k_out[i] = __half2float(k_src[i]);
        v_out[i] = __half2float(v_src[i]);
    }
    __syncthreads();
}

__device__ void kv_cache_rollback(KVCache& cache, int target_len) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *cache.seq_len = target_len;
    __syncthreads();
}
