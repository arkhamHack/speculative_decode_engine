#pragma once
#include "config.h"
#include "kv_cache.h"
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

// Tiled causal flash-attention over a paged KV cache (GQA-aware).
//
// Computes attn_out[d_model] = concat_h( softmax(Q_h @ K^T / sqrt(dph)) @ V )
// where Q_h uses head h and K/V use head (h * nkv / nh) for GQA.
//
// Caller responsibilities:
//   - q_all[d] must already contain Q with RoPE applied.
//   - attn_out[d] must be zeroed before the call.
//   - blk_logits must point to KV_BLOCK_SIZE floats of shared-memory scratch.
//   - The five shared scalars must be declared __shared__ by the caller.
//
// The function is cooperative across the thread block (uses __syncthreads).

__device__ inline void device_flash_attention_gqa(
    const float* __restrict__ q_all,
    float*       __restrict__ attn_out,
    const KVCache& kv, int layer,
    int nh, int nkv, int dph,
    int total_len,
    float* blk_logits,
    float& s_m, float& s_l,
    float& s_tm, float& s_alpha, float& s_inv)
{
    const int tid   = threadIdx.x;
    const float scale = rsqrtf((float)dph);
    const int n_logical_blocks = (total_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    for (int h = 0; h < nh; h++) {
        const int q_off   = h * dph;
        const int kv_head = h * nkv / nh;
        const int kv_off  = kv_head * dph;

        if (tid == 0) { s_m = -FLT_MAX; s_l = 0.f; }
        for (int ei = tid; ei < dph; ei += blockDim.x)
            attn_out[q_off + ei] = 0.f;
        __syncthreads();

        for (int log_blk = 0; log_blk < n_logical_blocks; log_blk++) {
            const int p0 = log_blk * KV_BLOCK_SIZE;

            // Q @ K^T for this tile
            for (int sl = tid; sl < KV_BLOCK_SIZE; sl += blockDim.x) {
                int p       = p0 + sl;
                float logit = -FLT_MAX;
                if (p < total_len) {
                    int bidx  = p / KV_BLOCK_SIZE;
                    int sl_kv = p % KV_BLOCK_SIZE;
                    int pb    = kv.layers[layer].block_table[bidx];
                    const half* base =
                        kv.pool +
                        (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* k_src = base + sl_kv * kv.d_head + kv_off;
                    float dot = 0.f;
                    for (int e = 0; e < dph; e++)
                        dot += q_all[q_off + e] * __half2float(k_src[e]);
                    logit = dot * scale;
                }
                blk_logits[sl] = logit;
            }
            __syncthreads();

            // Online softmax: update running max and denominator (tid 0 only)
            if (tid == 0) {
                float m_tile = -FLT_MAX;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++)
                    if (p0 + sl < total_len)
                        m_tile = fmaxf(m_tile, blk_logits[sl]);
                float m_prev = s_m;
                float m_new  = fmaxf(m_prev, m_tile);
                float alpha_u = (log_blk > 0) ? expf(m_prev - m_new) : 1.f;
                float l_tile = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    if (p0 + sl >= total_len) continue;
                    l_tile += expf(blk_logits[sl] - m_new);
                }
                s_m     = m_new;
                s_l     = alpha_u * s_l + l_tile;
                s_tm    = m_new;
                s_alpha = alpha_u;
            }
            __syncthreads();

            // Rescale prior accumulation
            float rescale = s_alpha;
            for (int ei = tid; ei < dph; ei += blockDim.x)
                attn_out[q_off + ei] *= rescale;
            __syncthreads();

            // Weighted V accumulation for this tile
            float m_here = s_tm;
            for (int ei = tid; ei < dph; ei += blockDim.x) {
                float acc_c = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    int p = p0 + sl;
                    if (p >= total_len) continue;
                    float w    = expf(blk_logits[sl] - m_here);
                    int bidx   = p / KV_BLOCK_SIZE;
                    int sl_kv  = p % KV_BLOCK_SIZE;
                    int pb     = kv.layers[layer].block_table[bidx];
                    const half* base =
                        kv.pool +
                        (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* v_src =
                        base + KV_BLOCK_SIZE * kv.d_head + sl_kv * kv.d_head;
                    acc_c += w * __half2float(v_src[kv_off + ei]);
                }
                attn_out[q_off + ei] += acc_c;
            }
            __syncthreads();
        }

        // Final normalization for this head
        if (tid == 0)
            s_inv = (s_l > 1e-20f && isfinite(s_l)) ? (1.f / s_l) : 0.f;
        __syncthreads();

        float norm_den = s_inv;
        for (int ei = tid; ei < dph; ei += blockDim.x)
            attn_out[q_off + ei] *= norm_den;
        __syncthreads();
    }
}

// Attention dispatch — routes to the correct attention backend based on type.
//
// Currently only ATTN_SOFTMAX is implemented.  Adding a new backend requires:
//   1. Add a case to the enum in config.h
//   2. Implement a device_*_attention() function in this header (or a new one)
//   3. Add a case to this switch
//   4. Update model_load_weights to parse the attention type from the header
//   5. Update export_model.py to write the attention type

__device__ inline void device_attention_dispatch(
    AttentionType attn_type,
    const float* __restrict__ q_all,
    float*       __restrict__ attn_out,
    const KVCache& kv, int layer,
    int nh, int nkv, int dph,
    int total_len,
    float* blk_logits,
    float& s_m, float& s_l,
    float& s_tm, float& s_alpha, float& s_inv)
{
    switch (attn_type) {
    case ATTN_SOFTMAX:
        device_flash_attention_gqa(
            q_all, attn_out, kv, layer,
            nh, nkv, dph, total_len,
            blk_logits, s_m, s_l, s_tm, s_alpha, s_inv);
        break;
    default:
        if (threadIdx.x == 0)
            printf("[FATAL] Unsupported attention type %d at layer %d\n",
                   (int)attn_type, layer);
        __trap();
    }
}
