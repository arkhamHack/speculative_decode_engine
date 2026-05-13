#pragma once
#include "config.h"
#include <cuda_fp16.h>

// ============================================================================
// Paged KV Cache Design
//
// Memory layout (per block):
//   K: half[KV_BLOCK_SIZE][d_head]   contiguous rows, then
//   V: half[KV_BLOCK_SIZE][d_head]   contiguous rows
//   Total per block = 2 * KV_BLOCK_SIZE * d_head * sizeof(half)
//
// Block table (per layer):
//   int block_table[MAX_KV_BLOCKS]  -- maps logical block index to physical
//   pool slot.  Physical slots are indices into a pre-allocated pool.
//
// Rollback:
//   Set seq_len = target_len.  No deallocation; blocks are reused on next
//   append (data overwritten in-place).
// ============================================================================

// Points to the raw storage of one KV block (K then V, contiguous).
// The total number of halfs = 2 * KV_BLOCK_SIZE * d_head.
struct KVBlock {
    half* data;   // device pointer, size = 2 * KV_BLOCK_SIZE * d_head
};

// Per-model, per-layer cache metadata kept on the device.
struct KVCacheLayer {
    int block_table[MAX_KV_BLOCKS]; // logical -> physical block index
    int n_blocks_used;              // how many logical blocks are allocated
};

// Full KV cache for one model (all layers).
struct KVCache {
    // Layer metadata (device memory, array of n_layers)
    KVCacheLayer* layers;

    // Physical block pool shared across all layers.
    // Pool contains pool_size blocks; each block has
    // 2 * KV_BLOCK_SIZE * d_head halfs.
    half*  pool;           // flat device buffer
    int    pool_size;      // total physical blocks
    int    d_head;         // total KV dimension stored per token (= d_model)
                           // per-head dim = d_head / n_heads (computed at runtime)
    int    n_layers;

    // Global sequence length for this cache. All layers share the same seq_len.
    int*   seq_len;        // device pointer to single int

    // Host-side pool allocator cursor (monotonically increasing).
    int    next_free_block;
};

// ============================================================================
// Host API
// ============================================================================

// Allocate a KV cache for a model on the GPU.
void kv_cache_alloc(KVCache& cache, int n_layers, int d_head,
                    int max_blocks_per_layer);

// Free all GPU memory associated with the cache.
void kv_cache_free(KVCache& cache);

// Reset sequence length to 0 and clear block tables (host call).
void kv_cache_reset(KVCache& cache);

// ============================================================================
// Device functions -- called from within kernels
// ============================================================================

// Append one token's K and V vectors to the cache for a specific layer.
// k_vec and v_vec are float arrays of length d_head (in registers/smem).
// seq_len is the *current* sequence length (before this append).
__device__ void kv_cache_append(KVCache& cache, int layer,
                                const float* k_vec, const float* v_vec,
                                int current_seq_len);

// Read K[pos] and V[pos] from the cache for a given layer into float arrays.
__device__ void kv_cache_read(const KVCache& cache, int layer, int pos,
                              float* k_out, float* v_out);

// Rollback: set seq_len to target_len (device-side, single thread).
__device__ void kv_cache_rollback(KVCache& cache, int target_len);
