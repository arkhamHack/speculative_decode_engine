"""
tools/export_model.py
Export a HuggingFace LLaMA-family / Qwen2 model to SDEC binary format.

Architecture requirements
-------------------------
The SDEC CUDA runtime implements a LLaMA-style transformer:
  - RMSNorm (not LayerNorm)
  - SwiGLU MLP (gate_proj * silu + up_proj, then down_proj)
  - Multi-head self-attention with RoPE on Q/K (Llama/HF layout)
  - Native GQA support (n_kv_heads <= n_heads; K/V stored at native rank)
  - Optional Q/K/V projection biases (Qwen2 / SDEC v5)

Supported HuggingFace model families
-------------------------------------
Verified:  LLaMA, Llama-2, Llama-3, Mistral, TinyLlama, SmolLM, Qwen2, Qwen2.5.
Not supported: GPT-2 (LayerNorm, GELU, combined QKV), BERT, T5, Phi, Gemma.

SDEC binary format (v5)
-----------------------
Header:
  4 bytes   magic          "SDEC"
  uint32    version        5
  uint32    n_layers
  uint32    d_model
  uint32    n_heads
  uint32    n_kv_heads
  uint32    d_ff
  uint32    vocab_size
  float32   rope_theta
  int32     rope_scaling_type    (0=none, 1=linear, 2=ntk, 3=yarn)
  float32   rope_scaling_factor  (1.0 if no scaling)
  int32     attention_type       (0=softmax, ...)
  int32     has_qkv_bias         (0/1)

Global / per-layer tensors as in v4. When has_qkv_bias=1, each layer also has:
  Wq_bias [d_model], Wk_bias [kv_dim], Wv_bias [kv_dim] after W_down.

Usage
-----
    python tools/export_model.py org/smaller-llama-model -o weights/draft.bin
    python tools/export_model.py Qwen/Qwen2.5-0.5B -o weights/qwen25_05b.sdec.bin
"""

import struct
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

LLAMA_FAMILY_ARCHS = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "SmolLMForCausalLM",
    "Qwen2ForCausalLM",
}

ROPE_SCALING_TYPES = {
    "linear": 1,
    "dynamic": 2,
    "ntk": 2,
    "yarn": 3,
}


def write_tensor(f, tensor: torch.Tensor, name: str = "") -> None:
    """Write element count (uint32) + float16 data."""
    arr = tensor.detach().to(torch.float16).cpu().numpy().astype(np.float16)
    n = arr.size
    f.write(struct.pack("I", n))
    f.write(arr.tobytes())
    kb = n * 2 / 1024
    if name:
        print(f"    {name:30s}  shape={tuple(tensor.shape)}  {kb:.1f} KB")


def parse_rope_scaling(cfg: dict) -> tuple[int, float]:
    scaling = cfg.get("rope_scaling")
    if scaling is None:
        return 0, 1.0
    rope_type = scaling.get("type", scaling.get("rope_type", ""))
    scaling_type = ROPE_SCALING_TYPES.get(rope_type, 0)
    scaling_factor = float(scaling.get("factor", 1.0))
    return scaling_type, scaling_factor


def load_state_dict(model_name: str) -> tuple[dict, dict]:
    """Load config + weights without AutoModel (avoids broken torchvision)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(model_name, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    files = list_repo_files(model_name)
    st_files = [p for p in files if p.endswith(".safetensors") and not p.endswith(".index.json")]
    # Prefer sharded model weights over tokenizer/extra files
    weight_files = [p for p in st_files if "model" in Path(p).name.lower() or p.startswith("model")]
    if not weight_files:
        weight_files = [p for p in st_files if "tokenizer" not in p.lower()]
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found for {model_name}")

    index_candidates = [p for p in files if p.endswith("model.safetensors.index.json")]
    sd = {}
    if index_candidates:
        idx_path = hf_hub_download(model_name, index_candidates[0])
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        for shard in shard_names:
            local = hf_hub_download(model_name, shard)
            print(f"  loading shard {shard}")
            sd.update(load_file(local))
    else:
        # Single-file or few files — load all candidate weight files
        for wf in sorted(set(weight_files)):
            local = hf_hub_download(model_name, wf)
            print(f"  loading {wf}")
            sd.update(load_file(local))

    return cfg, sd


def export_model(model_name: str, output_path: str) -> None:
    print(f"Downloading / loading: {model_name}")
    cfg, sd = load_state_dict(model_name)

    archs = cfg.get("architectures", [""])
    arch = archs[0] if archs else ""
    print(f"Architecture: {arch}")
    if arch not in LLAMA_FAMILY_ARCHS:
        print(f"WARNING: '{arch}' is not a verified LLaMA-family architecture. "
              f"Export may produce incorrect weights.")

    n_layers   = int(cfg["num_hidden_layers"])
    d_model    = int(cfg["hidden_size"])
    n_heads    = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    d_ff       = int(cfg["intermediate_size"])
    vocab_size = int(cfg["vocab_size"])
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    head_dim   = d_model // n_heads
    kv_dim     = n_kv_heads * head_dim

    rope_scaling_type, rope_scaling_factor = parse_rope_scaling(cfg)

    # Detect Q/K/V biases (Qwen2)
    has_qkv_bias = any(
        f"model.layers.{l}.self_attn.q_proj.bias" in sd
        for l in range(n_layers)
    )

    print(f"layers={n_layers} d={d_model} heads={n_heads} kv_heads={n_kv_heads} "
          f"d_ff={d_ff} vocab={vocab_size} rope_theta={rope_theta}")
    print(f"rope_scaling: type={rope_scaling_type} factor={rope_scaling_factor}")
    print(f"Native GQA: head_dim={head_dim} kv_dim={kv_dim}")
    print(f"QKV bias: {'yes' if has_qkv_bias else 'no'}")

    def W(key):
        return sd[key].T.contiguous()

    def bias_or_zeros(key, n):
        if key in sd:
            return sd[key].contiguous()
        return torch.zeros(n, dtype=torch.float16)

    attention_type = 0  # ATTN_SOFTMAX

    print(f"\nWriting SDEC v5 → {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"SDEC")
        f.write(struct.pack(
            "IIIIIIIfifii",
            5, n_layers, d_model, n_heads,
            n_kv_heads, d_ff, vocab_size,
            rope_theta,
            rope_scaling_type, rope_scaling_factor,
            attention_type,
            1 if has_qkv_bias else 0,
        ))

        write_tensor(f, sd["model.embed_tokens.weight"], "token_embedding")
        write_tensor(f, sd["model.norm.weight"], "rms_final_weight")

        if "lm_head.weight" in sd:
            lm_w = sd["lm_head.weight"]
        else:
            print("    (tie_word_embeddings: reusing embed_tokens for lm_head)")
            lm_w = sd["model.embed_tokens.weight"]
        write_tensor(f, lm_w.T.contiguous(), "output_proj")

        for l in range(n_layers):
            p = f"model.layers.{l}"
            print(f"  Layer {l}:")
            write_tensor(f, sd[f"{p}.input_layernorm.weight"], "rms_attn")
            write_tensor(f, W(f"{p}.self_attn.q_proj.weight"), "Wq")
            write_tensor(f, W(f"{p}.self_attn.k_proj.weight"), "Wk")
            write_tensor(f, W(f"{p}.self_attn.v_proj.weight"), "Wv")
            write_tensor(f, W(f"{p}.self_attn.o_proj.weight"), "Wo")
            write_tensor(f, sd[f"{p}.post_attention_layernorm.weight"], "rms_mlp")
            write_tensor(f, W(f"{p}.mlp.gate_proj.weight"), "W_gate")
            write_tensor(f, W(f"{p}.mlp.up_proj.weight"), "W_up")
            write_tensor(f, W(f"{p}.mlp.down_proj.weight"), "W_down")
            if has_qkv_bias:
                write_tensor(f, bias_or_zeros(f"{p}.self_attn.q_proj.bias", d_model), "Wq_bias")
                write_tensor(f, bias_or_zeros(f"{p}.self_attn.k_proj.bias", kv_dim), "Wk_bias")
                write_tensor(f, bias_or_zeros(f"{p}.self_attn.v_proj.bias", kv_dim), "Wv_bias")

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\nSaved: {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace LLaMA-family / Qwen2 model to SDEC binary"
    )
    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument("--output", "-o", default="model.bin",
                        help="Output .bin file path (default: model.bin)")
    args = parser.parse_args()

    try:
        export_model(args.model, args.output)
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        print("Install with:  pip install transformers torch safetensors huggingface_hub")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Expected weight key not found: {e}")
        print("This model may not be a standard LLaMA-family architecture.")
        sys.exit(1)


if __name__ == "__main__":
    main()
