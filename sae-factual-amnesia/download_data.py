"""
download_data.py — Run once to cache datasets locally
=======================================================
Saves TruthfulQA (multiple-choice) and WikiText to disk so
experiment runs don't re-download from HuggingFace every time.

Usage:
    python download_data.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")
SEED     = 42


def download_truthful_qa(out_path: Path):
    """
    Download TruthfulQA and reformat into 4-choice MC items.

    TruthfulQA's multiple_choice split provides mc1_targets (single correct
    answer). We use mc1 since it maps cleanly to a single correct letter,
    which is what the log-prob scorer in experiment.py expects.

    Each saved item:
        { "question": str, "choices": [A, B, C, D], "label": int (0-3) }
    """
    print("[download] Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    items = []
    rng   = random.Random(SEED)

    for row in dataset:
        mc      = row["mc1_targets"]
        choices = list(mc["choices"])
        labels  = list(mc["labels"])
        correct = labels.index(1)

        # Pad to exactly 4 choices if needed (rare edge case)
        wrong = [c for i, c in enumerate(choices) if i != correct]
        while len(choices) < 4:
            choices.append(rng.choice(wrong))
            labels.append(0)
        choices = choices[:4]
        labels  = labels[:4]

        # Shuffle so the correct answer isn't always position 0
        combined = list(zip(choices, labels))
        rng.shuffle(combined)
        choices, labels = zip(*combined)
        correct = list(labels).index(1)

        items.append({
            "question": row["question"],
            "choices":  list(choices),
            "label":    correct,
        })

    out_path.write_text(json.dumps(items, indent=2))
    print(f"  Saved {len(items)} items to {out_path}")


def download_wikitext(n: int, out_path: Path):
    """Save n non-empty WikiText-103 paragraphs for perplexity measurement."""
    print(f"[download] Loading WikiText-103 ({n} paragraphs)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= n:
            break
        if len(item["text"].strip()) > 50:
            texts.append(item["text"])

    out_path.write_text(json.dumps(texts, indent=2))
    print(f"  Saved {len(texts)} paragraphs to {out_path}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    download_truthful_qa(DATA_DIR / "truthful_qa.json")
    download_wikitext(200, DATA_DIR / "wikitext.json")
    print("\n[download] Done. Run 'python main.py' to start the experiment.")


if __name__ == "__main__":
    main()