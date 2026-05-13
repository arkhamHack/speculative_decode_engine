"""
tools/tokenize.py
Convert text ↔ token ID binary files for use with spec_decode.exe.

Binary (.tok) format
--------------------
  [n_tokens : uint32]
  [token_ids: n_tokens * uint32]

This matches the format read/written by tokenizer.h / tokenizer.cu.

Commands
--------
encode  -- text → .tok file (use before running spec_decode)
decode  -- .tok file → text (use after running spec_decode to read output)
inspect -- print token ids and their string representations

Usage examples
--------------
    # Tokenize a prompt
    python tools/tokenize.py encode TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
           "Once upon a time" prompt.tok

    # Run benchmark
    spec_decode.exe --draft=draft.bin --target=target.bin \\
                    --prompt-tok=prompt.tok --output-tok=output.tok

    # Decode the generated token ids back to text
    python tools/tokenize.py decode TinyLlama/TinyLlama-1.1B-Chat-v1.0 output.tok

    # Inspect what tokens a .tok file contains
    python tools/tokenize.py inspect TinyLlama/TinyLlama-1.1B-Chat-v1.0 prompt.tok
"""

import struct
import sys
import argparse
from pathlib import Path


def write_tok(path: str, ids: list[int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(ids)))
        for i in ids:
            f.write(struct.pack("I", int(i)))


def read_tok(path: str) -> list[int]:
    with open(path, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        ids = [struct.unpack("I", f.read(4))[0] for _ in range(n)]
    return ids


def cmd_encode(args) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    ids = tok.encode(args.text, add_special_tokens=True)
    write_tok(args.output, ids)

    print(f"Encoded {len(ids)} tokens → {args.output}")
    print(f"Tokens: {ids}")
    if len(ids) <= 32:
        decoded = [tok.decode([i]) for i in ids]
        print(f"Pieces: {decoded}")


def cmd_decode(args) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    ids = read_tok(args.input)
    text = tok.decode(ids, skip_special_tokens=True)

    print(f"Decoded {len(ids)} tokens:")
    print(text)


def cmd_inspect(args) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    ids = read_tok(args.input)
    print(f"\n{Path(args.input).name}: {len(ids)} tokens\n")
    print(f"{'idx':>5}  {'id':>7}  piece")
    print("-" * 40)
    for i, tid in enumerate(ids):
        piece = repr(tok.decode([tid]))
        print(f"{i:>5}  {tid:>7}  {piece}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text for spec_decode or decode its output"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # encode
    p_enc = sub.add_parser("encode", help="text → .tok binary file")
    p_enc.add_argument("model",  help="HuggingFace model name or local path")
    p_enc.add_argument("text",   help="Text to tokenize")
    p_enc.add_argument("output", help="Output .tok file")

    # decode
    p_dec = sub.add_parser("decode", help=".tok binary file → text")
    p_dec.add_argument("model", help="HuggingFace model name or local path")
    p_dec.add_argument("input", help="Input .tok file")

    # inspect
    p_ins = sub.add_parser("inspect", help="print token ids and string pieces")
    p_ins.add_argument("model", help="HuggingFace model name or local path")
    p_ins.add_argument("input", help="Input .tok file")

    args = parser.parse_args()

    try:
        if args.cmd == "encode":
            cmd_encode(args)
        elif args.cmd == "decode":
            cmd_decode(args)
        elif args.cmd == "inspect":
            cmd_inspect(args)
    except ImportError:
        print("ERROR: transformers library not found.")
        print("Install with:  pip install transformers")
        sys.exit(1)


if __name__ == "__main__":
    main()
