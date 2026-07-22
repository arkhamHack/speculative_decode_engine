"""Compare CUDA output.tok vs HuggingFace greedy on same prompt."""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_tok(p: Path) -> list[int]:
    with open(p, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        return [struct.unpack("I", f.read(4))[0] for _ in range(n)]


def read_sdec_rope(path: Path) -> float:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"SDEC":
            return float("nan")
        _ver, *_rest = struct.unpack("IIIIII", f.read(24))
        return struct.unpack("<f", f.read(4))[0]


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B"
    prompt_path = ROOT / "prompt.tok"
    output_path = ROOT / "output.tok"
    weights_path = ROOT / "weights" / "llama32_1b.sdec.bin"

    prompt_ids = read_tok(prompt_path)
    cuda_out = read_tok(output_path)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"SDEC rope_theta in bin: {read_sdec_rope(weights_path)}")
    print(f"Prompt ({len(prompt_ids)} tok): {tok.decode(prompt_ids)!r}")
    print(f"CUDA continuation: {tok.decode(cuda_out, skip_special_tokens=True)!r}")
    print(f"CUDA full text: {tok.decode(prompt_ids + cuda_out, skip_special_tokens=True)!r}")
    print()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    print(f"HF config rope_theta: {getattr(cfg, 'rope_theta', None)}")
    print(f"HF config rope_scaling: {getattr(cfg, 'rope_scaling', None)}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([prompt_ids])
        for _ in range(len(cuda_out)):
            logits = model(inp).logits[:, -1, :]
            nxt = logits.argmax(dim=-1)
            inp = torch.cat([inp, nxt.unsqueeze(0)], dim=1)
    hf_new = inp[0, len(prompt_ids) :].tolist()
    print(f"HF matches CUDA token-for-token: {hf_new == cuda_out}")
    if hf_new != cuda_out:
        for i, (a, b) in enumerate(zip(hf_new, cuda_out)):
            if a != b:
                print(f"  first mismatch @ {i}: HF={a} CUDA={b}")
                print(f"    HF piece: {tok.decode([a])!r}  CUDA piece: {tok.decode([b])!r}")
                break
    print(f"HF continuation: {tok.decode(hf_new, skip_special_tokens=True)!r}")
    print(f"HF full text: {tok.decode(prompt_ids + hf_new, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
