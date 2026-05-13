---
name: CUDA Speculative Decode Port
overview: Port the speculative decoding engine to C++/CUDA with a persistent megakernel, paged GPU-resident KV cache, zero CPU-GPU sync during generation, and a pybind11 Python CLI wrapper. Both a multi-kernel baseline and persistent megakernel path will be implemented, switchable by flag.
todos:
  - id: config-header
    content: Create include/config.h with model dimensions, block sizes, max sequence length constants
    status: completed
  - id: utils
    content: "Create include/utils.h + src/utils.cu with device math: warp_reduce_max, warp_reduce_sum, block_softmax, rmsnorm, vectorized half2/float4 load helpers"
    status: completed
  - id: kv-cache
    content: "Create include/kv_cache.h + src/kv_cache.cu: KVBlock struct, BlockTable, device append/read/rollback, host pool allocation"
    status: completed
  - id: model
    content: "Create include/model.h + src/model.cu: ModelWeights struct, device functions for attention (with KV cache interaction), SwiGLU MLP, full transformer layer, full forward pass (multi-layer)"
    status: completed
  - id: kernels-multi
    content: "Create include/kernels.h + src/kernels.cu: multi-kernel path with separate launches for rmsnorm, attention, mlp, argmax, and speculative verify"
    status: completed
  - id: kernels-mega
    content: "Add persistent megakernel to src/kernels.cu: single cooperative-groups kernel with generation loop, draft/verify/accept phases, grid-level barriers"
    status: completed
  - id: host-main
    content: "Create src/main.cu: host entry point with weight initialization, memory allocation, kernel dispatch (multi vs mega), result readback, timing"
    status: completed
  - id: pybind
    content: Create src/binding.cu with pybind11 module exposing run_benchmark(mode, max_tokens, k, seed) returning results dict
    status: completed
  - id: build
    content: Create Makefile and CMakeLists.txt with targets for the library, pybind11 module, and standalone executable
    status: completed
  - id: python-cli
    content: "Create cli/benchmark.py: Python CLI that imports the pybind11 module, runs baseline vs speculative, prints comparison metrics"
    status: completed
  - id: test-verify
    content: "Test: compile, run baseline and speculative, verify identical greedy output, print timing comparison"
    status: in_progress
isProject: false
---

# C++/CUDA Speculative Decoding Engine

## Architecture Overview

```mermaid
graph TD
    subgraph python_cli [Python CLI via pybind11]
        CLI["cli/benchmark.py"]
        Bind["pybind11 module"]
    end
    subgraph host_code [C++ Host Code]
        Main["main.cu - init, alloc, launch"]
        ModelInit["model.cu - weight init, configs"]
        KVInit["kv_cache.cu - block pool alloc"]
    end
    subgraph gpu_kernel [GPU Kernel Layer]
        MultiK["Multi-kernel path: one launch per op"]
        MegaK["Persistent megakernel: single launch, loop inside"]
        Attn["Attention + KV append"]
        MLP["SwiGLU MLP"]
        Norm["RMSNorm"]
        Spec["Speculative verify + accept/reject"]
    end
    subgraph gpu_memory [GPU Memory]
        Weights["Model weights - float16, aligned"]
        KVPool["Paged KV block pool"]
        BlockTable["Block table: seq -> blocks"]
        Activations["Activation buffers"]
    end

    CLI --> Bind --> Main
    Main --> MultiK
    Main --> MegaK
    MultiK --> Attn & MLP & Norm & Spec
    MegaK --> Attn & MLP & Norm & Spec
    Attn --> KVPool
    Attn --> BlockTable
    MLP --> Weights
    Norm --> Weights
```

## Model Design

Two minimal transformers sharing the same architecture but different sizes:

- **Draft model**: 2 layers, d_model=128, 1 head, d_head=128
- **Target model**: 4 layers, d_model=256, 1 head, d_head=256

Each layer: RMSNorm -> Single-Head Attention -> RMSNorm -> SwiGLU MLP

Weights randomly initialized with small stddev. Vocab size = 256 (dummy integer tokenizer). All weights in float16 with 16-byte alignment for vectorized `float4` loads.

## KV Cache -- Paged Block Design

```mermaid
graph LR
    subgraph block_table [Block Table per layer]
        BT0["Block ptr 0"]
        BT1["Block ptr 1"]
        BT2["Block ptr 2"]
        BTn["..."]
    end
    subgraph physical_blocks [Physical Block Pool]
        B0["Block 0: K[16,d] V[16,d]"]
        B1["Block 1: K[16,d] V[16,d]"]
        B2["Block 2: K[16,d] V[16,d]"]
    end
    BT0 --> B2
    BT1 --> B0
    BT2 --> B1
```

- Block size: 16 tokens
- Each block stores `K[16][d_head]` and `V[16][d_head]` in float16 contiguously
- Memory layout per block: `[K_row0, K_row1, ..., K_row15, V_row0, ..., V_row15]`
- Block table: `int block_table[MAX_LAYERS][MAX_BLOCKS]` mapping logical block index to physical pool slot
- `seq_len` counter per sequence tracks current fill position
- **Append**: compute `block_idx = seq_len / BLOCK_SIZE`, `slot = seq_len % BLOCK_SIZE`, write K/V
- **Rollback**: set `seq_len = target_len`, no deallocation needed (blocks reused on next append)
- Pool pre-allocated at init; no dynamic allocation during generation

## Speculative Decoding Loop (GPU-resident)

```mermaid
sequenceDiagram
    participant DraftModel
    participant TargetModel
    participant KVCache
    participant Output

    loop while not done
        Note over DraftModel: Draft k tokens autoregressively
        DraftModel->>KVCache: Append draft K/V (draft cache)
        DraftModel->>DraftModel: argmax -> draft_ids[0..k-1]

        Note over TargetModel: Verify all k tokens in one pass
        TargetModel->>KVCache: Append [last_tok, draft_ids] (target cache)
        TargetModel->>TargetModel: argmax at each position -> target_ids[0..k]

        Note over Output: Accept longest matching prefix
        alt target_ids[i] == draft_ids[i] for all i
            Output->>Output: Accept all k + bonus token
        else mismatch at position n
            Output->>Output: Accept draft[0..n-1] + target[n]
            KVCache->>KVCache: Rollback both caches
        end
    end
```

Greedy mode only for the CUDA implementation (matching the greedy fast-path from the Python code). The accept/reject is a simple argmax comparison loop.

## File Structure

```
cuda/
  include/
    config.h          Model configs, constants (dims, block size, max seq len)
    kv_cache.h        KVBlock struct, BlockTable, device functions
    model.h           ModelWeights struct, layer configs
    utils.h           Device math: softmax, rmsnorm, warp reduce
    kernels.h         Kernel launch wrappers + megakernel entry
  src/
    main.cu           Host: alloc, init weights, launch, read results
    kv_cache.cu       Host pool alloc + device append/rollback functions
    model.cu          Device: attention, swiglu_mlp, rmsnorm, full_layer
    utils.cu          Device: warp_reduce, block_softmax, vectorized loads
    kernels.cu        Multi-kernel launches + persistent megakernel
    binding.cu        pybind11 module exposing run_benchmark()
  Makefile
  CMakeLists.txt
cli/
  benchmark.py        Python CLI calling the pybind11 module
```

## Kernel Design Detail

### Multi-kernel path (flag: `--mode=multi`)
Each generation step launches separate kernels:
- `rmsnorm_kernel` -- one block per row
- `attention_kernel` -- one warp per query position, iterates over KV blocks
- `mlp_kernel` -- one block per row, SwiGLU in shared memory
- `argmax_kernel` -- reduction to find next token
- `speculative_verify_kernel` -- compare draft vs target argmax arrays

### Persistent megakernel path (flag: `--mode=mega`)
Single kernel launched once with cooperative groups:
- Grid: enough blocks to cover one layer's computation
- Main thread (block 0, thread 0) runs the generation control loop
- Barrier via `cooperative_groups::grid::sync()` between phases
- All threads participate in matmul/attention/MLP as needed
- Shared memory reused across phases (double-buffered)
- Completion signaled via device-mapped host memory (zero-copy flag)

Key warp-level primitives used:
- `__shfl_xor_sync` for warp-level reductions (softmax, rmsnorm)
- `__shfl_sync` for broadcasting argmax results
- Vectorized `float4` / `half2` loads for weight reads

## Implementation Order and Dependencies

The implementation proceeds bottom-up: device math utilities first, then model layers, then KV cache, then kernels, then host orchestration, then pybind11 binding.

## Testing

- Dummy tokenizer: tokens are integers 0-255
- Initialize weights with small random values (deterministic seed)
- Run baseline (target-only) for 32 tokens
- Run speculative (draft+target) for 32 tokens
- Print both token sequences
- Verify identical output (greedy mode guarantees this)
- Print timing: tokens/sec, total time, acceptance rate
