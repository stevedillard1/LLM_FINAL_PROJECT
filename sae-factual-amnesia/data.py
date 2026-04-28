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

# MODEL_ID    = "google/gemma-3-1b-it"
# SAE_RELEASE = "gemma-scope-2-1b-it-resid_post"
# SAE_HOOK    = "blocks.12.hook_resid_post"
# SAE_ID      = "layer_12_width_16k_l0_small"

MODEL_ID    = "EleutherAI/pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
SAE_HOOK    = "blocks.3.hook_resid_post"
SAE_ID      = SAE_HOOK

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data")

# Local cache paths written by download_data.py
TRUTHFUL_QA_PATH  = DATA_DIR / "truthful_qa.json"
WIKITEXT_PATH     = DATA_DIR / "wikitext.json"


def _check_data_exists():
    missing = [p for p in [TRUTHFUL_QA_PATH, WIKITEXT_PATH] if not p.exists()]
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
    Load Gemma-3-1B-IT via TransformerLens and the matching Gemma Scope SAE.
    Returns (model, sae, tokenizer).

    Model weights (~2.5GB in float16) are cached by HuggingFace after first
    download at ~/.cache/huggingface.
    """
    print(f"[data] Loading model '{MODEL_ID}' on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        dtype=torch.float32,   # SAE encode/decode expects float32
    )
    model.eval()
    print("[data] Model loaded OK.")

    print(f"[data] Loading SAE layer '{SAE_ID}'...")
    try:
        sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_HOOK,
        device=DEVICE,
    )
    except Exception as e:
        print(f"[data] ERROR loading SAE: {type(e).__name__}: {e}")
        raise
    sae.eval()
    print(f"[data] SAE loaded OK. Feature dim: {sae.cfg.d_sae}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, sae, tokenizer


# ---------------------------------------------------------------------------
# Datasets — read from local files cached by download_data.py
# ---------------------------------------------------------------------------

def load_truthful_qa(n: int) -> list[dict]:
    """
    Load up to n TruthfulQA multiple-choice items from local cache.

    Each item has:
        item["question"]  — the question string
        item["choices"]   — list of 4 answer strings (A/B/C/D)
        item["label"]     — int index of the correct choice (0-3)

    Used for both feature profiling and evaluation.
    """
    _check_data_exists()
    items = json.loads(TRUTHFUL_QA_PATH.read_text())[:n]
    print(f"[data] Loaded {len(items)} TruthfulQA items from {TRUTHFUL_QA_PATH}")
    return items


def load_wikitext(n: int) -> list[str]:
    """
    Load up to n WikiText-103 paragraphs from local cache.
    Used for perplexity measurement (language quality check).
    """
    _check_data_exists()
    texts = json.loads(WIKITEXT_PATH.read_text())[:n]
    print(f"[data] Loaded {len(texts)} WikiText paragraphs from {WIKITEXT_PATH}")
    return texts