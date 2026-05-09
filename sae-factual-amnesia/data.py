"""
data.py — Model, SAE, and dataset loading
==========================================
All I/O lives here. experiment.py never touches datasets or model files directly.

Model and SAE are configured in config.yaml.
Run load_config() before anything else in main.py.
"""

import json
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformers import AutoTokenizer

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data")

TRUTHFUL_QA_PATH = DATA_DIR / "truthful_qa.json"
WIKITEXT_PATH    = DATA_DIR / "wikitext.json"

# Set by load_config() — do not edit these directly, edit config.yaml instead
MODEL_ID    = None
SAE_RELEASE = None
SAE_HOOK    = None
SAE_ID      = None


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Read config.yaml and set module-level model constants.
    Must be called before load_model_and_sae().
    Returns the full config dict so main.py can read experiment parameters.
    """
    global MODEL_ID, SAE_RELEASE, SAE_HOOK, SAE_ID

    try:
        import yaml
    except ImportError:
        print("[data] ERROR: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)

    path = Path(config_path)
    if not path.exists():
        print(f"[data] ERROR: config file not found at '{config_path}'")
        sys.exit(1)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    for field in ("model_id", "sae_release", "sae_hook", "sae_id"):
        if field not in cfg:
            print(f"[data] ERROR: '{field}' missing from {config_path}")
            sys.exit(1)

    MODEL_ID    = cfg["model_id"]
    SAE_RELEASE = cfg["sae_release"]
    SAE_HOOK    = cfg["sae_hook"]
    SAE_ID      = cfg["sae_id"]

    print(f"[data] Config loaded: model='{MODEL_ID}'")
    print(f"[data] SAE: {SAE_RELEASE} / {SAE_ID}")
    return cfg


def _check_data_exists():
    missing = [p for p in [TRUTHFUL_QA_PATH, WIKITEXT_PATH] if not p.exists()]
    if missing:
        print("[data] ERROR: Local data files not found:")
        for p in missing:
            print(f"  {p}")
        print("[data] Run 'python download_data.py' first.")
        sys.exit(1)


def load_model_and_sae():
    """
    Load model via TransformerLens and the matching pretrained SAE.
    Requires load_config() to have been called first.
    Returns (model, sae, tokenizer).
    """
    if MODEL_ID is None:
        print("[data] ERROR: load_config() must be called before load_model_and_sae().")
        sys.exit(1)

    print(f"[data] DEVICE = {DEVICE}")
    print(f"[data] Loading model '{MODEL_ID}'...")

    # Use float16 for large models to fit in VRAM, float32 for smaller ones
    FLOAT16_MODELS = {"google/gemma-2-4b-it", "google/gemma-2-9b-it"}
    dtype = torch.float16 if MODEL_ID in FLOAT16_MODELS else torch.float32

    model = HookedTransformer.from_pretrained(MODEL_ID, device="cpu", dtype=dtype)
    model = model.to(DEVICE)
    model.eval()
    print(f"[data] Model loaded OK (dtype={dtype}).")

    # model = HookedTransformer.from_pretrained(MODEL_ID, device="cpu", dtype=torch.float32)
    # model = model.to(DEVICE)
    # model.eval()
    # print("[data] Model loaded OK.")

    print(f"[data] Loading SAE '{SAE_ID}'...")
    try:
        sae, _, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=SAE_ID,
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


def load_truthful_qa(n: int) -> list[dict]:
    """
    Load up to n TruthfulQA MC items from local cache.
    TruthfulQA has 817 questions total.
    Each item: { "question": str, "choices": [A,B,C,D], "label": int }
    """
    _check_data_exists()
    items = json.loads(TRUTHFUL_QA_PATH.read_text())[:n]
    print(f"[data] Loaded {len(items)} TruthfulQA items from {TRUTHFUL_QA_PATH}")
    return items


def load_wikitext(n: int) -> list[str]:
    """Load up to n WikiText-103 paragraphs for perplexity measurement."""
    _check_data_exists()
    texts = json.loads(WIKITEXT_PATH.read_text())[:n]
    print(f"[data] Loaded {len(texts)} WikiText paragraphs from {WIKITEXT_PATH}")
    return texts