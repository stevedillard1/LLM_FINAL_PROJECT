"""
download_data.py — Run this once to cache datasets locally
===========================================================
Saves TriviaQA and WikiText to disk so experiment runs don't
re-download from HuggingFace every time.

Usage:
    python download_data.py

    # Optional: control how many samples to save
    python download_data.py --n-trivia-train 1000 --n-trivia-val 500 --n-wiki 200

Output files (in ./data/):
    data/trivia_train.json   — for feature profiling
    data/trivia_val.json     — for evaluation
    data/wikitext.json       — for perplexity
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("data")


def download_trivia_qa(split: str, n: int, out_path: Path):
    print(f"[download] TriviaQA {split} ({n} samples) → {out_path}")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split, streaming=True)

    items = []
    for item in dataset:
        if len(items) >= n:
            break
        # Only keep the fields we actually use
        items.append({
            "question": item["question"],
            "answer": {
                "value":   item["answer"]["value"],
                "aliases": item["answer"]["aliases"],
            }
        })

    out_path.write_text(json.dumps(items, indent=2))
    print(f"  Saved {len(items)} items.")


def download_wikitext(n: int, out_path: Path):
    print(f"[download] WikiText-103 ({n} paragraphs) → {out_path}")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= n:
            break
        if len(item["text"].strip()) > 50:
            texts.append(item["text"])

    out_path.write_text(json.dumps(texts, indent=2))
    print(f"  Saved {len(texts)} paragraphs.")


def main():
    p = argparse.ArgumentParser(description="Download and cache datasets locally")
    p.add_argument("--n-trivia-train", type=int, default=1000,
                   help="TriviaQA train samples (used for feature profiling).")
    p.add_argument("--n-trivia-val",   type=int, default=500,
                   help="TriviaQA validation samples (used for evaluation).")
    p.add_argument("--n-wiki",         type=int, default=200,
                   help="WikiText paragraphs (used for perplexity).")
    args = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    download_trivia_qa("train",      args.n_trivia_train, DATA_DIR / "trivia_train.json")
    download_trivia_qa("validation", args.n_trivia_val,   DATA_DIR / "trivia_val.json")
    download_wikitext(args.n_wiki,                        DATA_DIR / "wikitext.json")

    print("\n[download] Done. Run 'python main.py' to start the experiment.")


if __name__ == "__main__":
    main()
