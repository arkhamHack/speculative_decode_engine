"""
tools/export_model.py
Export a HuggingFace LLaMA-family model to SDEC binary format.

Architecture requirements
-------------------------
The SDEC CUDA runtime implements a LLaMA-style transformer:
  - RMSNorm (not LayerNorm)
  - SwiGLU MLP (gate_proj * silu + up_proj, then down_proj)
  - Multi-head self-attention with RoPE on Q/K (Llama/HF layout); KV cache stores rotated K.
  - Standard multi-head attention (n_kv_heads == n_heads or GQA is expanded)

Supported HuggingFace model families
-------------------------------------
Verified:  LLaMA, Llama-2, Llama-3, Mistral, TinyLlama, SmolLM,
           OPT-like LLaMA clones.
Partially: Models with GQA (SmolLM2, Mistral) — KV heads are expanded
           to full rank automatically.
Not supported: GPT-2 (LayerNorm, GELU, combined QKV), BERT, T5, Phi, Gemma.

RoPE note
---------
Base frequency ``rope_theta`` is stored in the SDEC header (format v2).
Llama 3+ ``rope_scaling`` (YaRN / NTK) is not applied in CUDA — moderate-length
greedy decoding still tracks HF closely when ``rope_theta`` matches the checkpoint.

SDEC binary format
------------------
Header after magic ``SDEC``:
  uint32 version       (2 — current exporter)
  uint32 n_layers, d_model, n_heads, d_ff, vocab_size
  float32 rope_theta   (v2 only; v1 loaders default rope_theta to 10000)

Tensors (fixed order, each preceded by uint32 element count):
  token_embedding  [vocab_size * d_model]  float16
  rms_final_weight [d_model]               float16
  output_proj      [d_model * vocab_size]  float16
  For each layer l = 0..n_layers-1:
    rms_attn_weight [d_model]
    Wq              [d_model, d_model]  row-major (= q_proj.weight.T)
    Wk              [d_model, d_model]
    Wv              [d_model, d_model]
    Wo              [d_model, d_model]  (= o_proj.weight.T)
    rms_mlp_weight  [d_model]
    W_gate          [d_model, d_ff]
    W_up            [d_model, d_ff]
    W_down          [d_ff,    d_model]

Usage
-----
    # Smaller model → draft weights (fewer layers / faster propose step)
    python tools/export_model.py org/smaller-llama-model -o weights/draft.bin

    # Larger model → target weights (verification model)
    python tools/export_model.py org/larger-llama-model -o weights/target.bin

    # Requires: pip install transformers torch
    # Then run the CUDA benchmark or enable Production mode in the web UI.
"""

import struct
import sys
import argparse
from pathlib import Path

import torch
import numpy as np


def to_fp16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a PyTorch tensor to raw float16 bytes (host-order)."""
    return tensor.detach().to(torch.float16).cpu().numpy().astype(np.float16).tobytes()


def write_tensor(f, tensor: torch.Tensor, name: str = "") -> None:
    """Write element count (uint32) + float16 data."""
    arr = tensor.detach().to(torch.float16).cpu().numpy().astype(np.float16)
    n = arr.size
    f.write(struct.pack("I", n))
    f.write(arr.tobytes())
    kb = n * 2 / 1024
    if name:
        print(f"    {name:30s}  shape={tuple(tensor.shape)}  {kb:.1f} KB")


def expand_gqa(weight: torch.Tensor, n_heads: int, n_kv_heads: int,
               d_model: int) -> torch.Tensor:
    """Expand grouped-query attention K/V projections to full n_heads rank.

    weight: [d_model, n_kv_heads * d_head]  (already transposed to [d_in, d_out])
    Returns: [d_model, n_heads * d_head]
    """
    if n_kv_heads == n_heads:
        return weight
    ratio    = n_heads // n_kv_heads
    d_head   = d_model // n_heads
    # Reshape to [d_model, n_kv_heads, d_head], repeat, reshape back
    w = weight.reshape(d_model, n_kv_heads, d_head)
    w = w.repeat_interleave(ratio, dim=1)          # [d_model, n_heads, d_head]
    return w.reshape(d_model, n_heads * d_head)    # [d_model, d_model]


def export_llama(model_name: str, output_path: str) -> None:
    """Export a LLaMA-family HuggingFace model to SDEC binary."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading config from {model_name} ...")
    cfg = AutoConfig.from_pretrained(model_name)

    # Validate architecture
    arch = getattr(cfg, "architectures", [""])[0]
    print(f"Architecture: {arch}")

    n_layers   = cfg.num_hidden_layers
    d_model    = cfg.hidden_size
    n_heads    = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    d_ff       = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

    if n_kv_heads != n_heads:
        print(f"  GQA detected (n_heads={n_heads}, n_kv_heads={n_kv_heads})  "
              f"— KV will be expanded to full rank")

    # Memory estimate
    params_m = (vocab_size * d_model * 2 +                         # embed + lm_head
                n_layers * (4 * d_model * d_model + 3 * d_model * d_ff)) / 1e6
    print(f"Config: layers={n_layers}  d={d_model}  heads={n_heads}  "
          f"d_ff={d_ff}  vocab={vocab_size}  rope_theta={rope_theta}")
    print(f"Estimated float16 size: {params_m * 2:.0f} MB")

    print(f"\nLoading weights (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    sd = model.state_dict()

    # Helper to get weight with .T (PyTorch Linear stores [d_out, d_in])
    def W(key: str) -> torch.Tensor:
        return sd[key].T.contiguous()   # [d_in, d_out]

    print(f"\nWriting to {output_path} ...")
    with open(output_path, "wb") as f:
        # --- Header SDEC v2 (rope_theta for CUDA RoPE) ---
        f.write(b"SDEC")
        f.write(
            struct.pack(
                "IIIIII",
                2,
                n_layers,
                d_model,
                n_heads,
                d_ff,
                vocab_size,
            )
        )
        f.write(struct.pack("<f", rope_theta))

        # --- Global tensors ---
        print("  Global tensors:")
        write_tensor(f, sd["model.embed_tokens.weight"], "token_embedding")
        write_tensor(f, sd["model.norm.weight"],          "rms_final_weight")

        # lm_head may be tied to embed_tokens
        lm_key = "lm_head.weight"
        lm_w   = sd.get(lm_key, sd["model.embed_tokens.weight"])
        # lm_head.weight is [vocab, d] → transpose to [d, vocab] for SDEC
        write_tensor(f, lm_w.T.contiguous(), "output_proj")

        # --- Per-layer tensors ---
        for l in range(n_layers):
            print(f"  Layer {l}:")
            p = f"model.layers.{l}"

            write_tensor(f, sd[f"{p}.input_layernorm.weight"],      "rms_attn")

            Wq = W(f"{p}.self_attn.q_proj.weight")                  # [d, d]
            Wk = expand_gqa(
                W(f"{p}.self_attn.k_proj.weight"), n_heads, n_kv_heads, d_model)
            Wv = expand_gqa(
                W(f"{p}.self_attn.v_proj.weight"), n_heads, n_kv_heads, d_model)
            Wo = W(f"{p}.self_attn.o_proj.weight")                  # [d, d]

            write_tensor(f, Wq, "Wq")
            write_tensor(f, Wk, "Wk")
            write_tensor(f, Wv, "Wv")
            write_tensor(f, Wo, "Wo")

            write_tensor(f, sd[f"{p}.post_attention_layernorm.weight"], "rms_mlp")
            write_tensor(f, W(f"{p}.mlp.gate_proj.weight"),  "W_gate")
            write_tensor(f, W(f"{p}.mlp.up_proj.weight"),    "W_up")
            write_tensor(f, W(f"{p}.mlp.down_proj.weight"),  "W_down")

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\nSaved: {output_path}  ({size_mb:.1f} MB)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace LLaMA-family model to SDEC binary format"
    )
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument("--output", "-o", default="model.bin",
                        help="Output .bin file path (default: model.bin)")
    args = parser.parse_args()

    try:
        export_llama(args.model, args.output)
    except ImportError:
        print("ERROR: transformers library not found.")
        print("Install with:  pip install transformers torch")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Expected weight key not found: {e}")
        print("This model may not be a standard LLaMA-family architecture.")
        sys.exit(1)


if __name__ == "__main__":
    main()
