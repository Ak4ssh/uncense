#!/usr/bin/env python3
"""
Data preparation utility.
Converts various formats → clean JSONL for training.

Supported input formats:
  1. {"messages": [...]}                      (chat format)
  2. {"instruction": "...", "response": "..."} (alpaca format)
  3. {"prompt": "...", "response": "..."}      (simple pair)
  4. Raw text file — splits on double newlines

Usage:
  python prepare_data.py --input my_data.jsonl --output data/train.jsonl
  python prepare_data.py --input raw.txt --output data/train.jsonl --format text
  python prepare_data.py --input data.jsonl --split 0.9  # auto train/val split
"""

import argparse
import json
import random
from pathlib import Path


def convert_example(ex: dict) -> dict | None:
    """Normalize any known format to messages-based format."""
    if "messages" in ex:
        return ex  # Already correct format

    if "conversations" in ex:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        msgs = [{"role": role_map.get(m["from"], m["from"]), "content": m["value"]}
                for m in ex["conversations"]]
        return {"messages": msgs}

    if "instruction" in ex:
        msgs = [
            {"role": "user",      "content": ex["instruction"]},
            {"role": "assistant", "content": ex.get("response", ex.get("output", ""))},
        ]
        if ex.get("system"):
            msgs.insert(0, {"role": "system", "content": ex["system"]})
        return {"messages": msgs}

    if "prompt" in ex and "response" in ex:
        return {"messages": [
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]}

    return None  # Unknown format — skip


def process_jsonl(input_path: str) -> list[dict]:
    examples = []
    skipped  = 0
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                converted = convert_example(ex)
                if converted:
                    examples.append(converted)
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f"  Line {i}: JSON error — {e}")
                skipped += 1

    print(f"  Loaded: {len(examples)} examples  |  Skipped: {skipped}")
    return examples


def process_text(input_path: str, min_chars: int = 100) -> list[dict]:
    """Split plain text file on double newlines into Q&A pairs."""
    text = Path(input_path).read_text()
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) >= min_chars]
    examples = []
    for chunk in chunks:
        # Wrap each chunk as an assistant message (pre-training style)
        examples.append({"messages": [{"role": "assistant", "content": chunk}]})
    print(f"  Split text into {len(examples)} chunks")
    return examples


def write_jsonl(examples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  → Written: {path} ({len(examples)} examples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Input file path")
    parser.add_argument("--output",  default="data/train.jsonl", help="Output JSONL")
    parser.add_argument("--format",  default="jsonl", choices=["jsonl", "text"])
    parser.add_argument("--split",   type=float, default=None,
                        help="Train/val split ratio (e.g. 0.9 = 90%% train)")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print(f"\nLoading: {args.input}")
    if args.format == "jsonl":
        examples = process_jsonl(args.input)
    else:
        examples = process_text(args.input)

    if args.max_samples:
        examples = examples[:args.max_samples]
        print(f"  Capped at {args.max_samples} examples")

    if args.shuffle:
        random.shuffle(examples)
        print(f"  Shuffled")

    if args.split:
        cut  = int(len(examples) * args.split)
        train = examples[:cut]
        val   = examples[cut:]
        val_path = args.output.replace("train", "val")
        write_jsonl(train, args.output)
        write_jsonl(val,   val_path)
        print(f"\nSplit: {len(train)} train / {len(val)} val")
    else:
        write_jsonl(examples, args.output)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
