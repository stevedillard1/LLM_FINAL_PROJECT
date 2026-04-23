"""
data.py — Model, SAE, and dataset loading
==========================================
All I/O lives here. experiment.py never touches datasets or model files directly.

Datasets are loaded from local JSON files (written by download_data.py).
Run download_data.py once before running the experiment:
    python download_data.py
"""

import json
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants — edit these to swap model/layer
# ---------------------------------------------------------------------------

MODEL_ID    = "EleutherAI/pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
SAE_HOOK    = "blocks.3.hook_resid_post"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data")

# Local cache paths written by download_data.py
TRIVIA_TRAIN_PATH = DATA_DIR / "trivia_train.json"
TRIVIA_VAL_PATH   = DATA_DIR / "trivia_val.json"
WIKITEXT_PATH     = DATA_DIR / "wikitext.json"


def _check_data_exists():
    missing = [p for p in [TRIVIA_TRAIN_PATH, TRIVIA_VAL_PATH, WIKITEXT_PATH] if not p.exists()]
    if missing:
        print("[data] ERROR: Local data files not found:")
        for p in missing:
            print(f"  {p}")
        print("[data] Run 'python download_data.py' first to cache the datasets.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model + SAE
# ---------------------------------------------------------------------------

def load_model_and_sae():
    """
    Load Pythia-1.4B via TransformerLens and the matching pretrained SAE.
    Returns (model, sae, tokenizer).

    Note: model weights are cached by HuggingFace after the first download
    (~3 GB), so subsequent runs load from ~/.cache/huggingface.
    """
    print(f"[data] Loading model '{MODEL_ID}' on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE)
    model.eval()

    print("[data] Model loaded OK")        # add this

    print(f"[data] Loading SAE '{SAE_HOOK}'...")
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_HOOK,
        device=DEVICE,
    )
    sae.eval()

    print("[data] SAE loaded OK")          # add this

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[data] SAE feature dim: {sae.cfg.d_sae}")
    return model, sae, tokenizer


# ---------------------------------------------------------------------------
# Datasets — read from local files cached by download_data.py
# ---------------------------------------------------------------------------

def load_trivia_train(n: int) -> list[dict]:
    """
    Load up to n TriviaQA training items from local cache.
    Used for feature profiling in identify_factual_features().
    """
    _check_data_exists()
    items = json.loads(TRIVIA_TRAIN_PATH.read_text())[:n]
    print(f"[data] Loaded {len(items)} TriviaQA train items from {TRIVIA_TRAIN_PATH}")
    return items


def load_trivia_val(n: int) -> list[dict]:
    """
    Load up to n TriviaQA validation items from local cache.
    Used for evaluation.
    """
    _check_data_exists()
    items = json.loads(TRIVIA_VAL_PATH.read_text())[:n]
    print(f"[data] Loaded {len(items)} TriviaQA val items from {TRIVIA_VAL_PATH}")
    return items


def load_wikitext(n: int) -> list[str]:
    """
    Load up to n WikiText paragraphs from local cache.
    Used for perplexity measurement.
    """
    _check_data_exists()
    texts = json.loads(WIKITEXT_PATH.read_text())[:n]
    print(f"[data] Loaded {len(texts)} WikiText paragraphs from {WIKITEXT_PATH}")
    return texts
