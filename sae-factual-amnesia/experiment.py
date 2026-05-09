"""
experiment.py — Core experiment logic
======================================
Four conditions are compared:
  1. baseline          — no intervention
  2. targeted          — top-k factual SAE features suppressed
  3. random control    — equal number of randomly sampled features suppressed
  4. freq-matched ctrl — features matched by activation frequency to the factual set

The frequency-matched control is the stronger test: if random features are mostly
inactive, zeroing them does nothing and the control looks like baseline trivially.
Matching activation frequency ensures the control suppresses features that are
actually doing work, making a targeted vs control gap more meaningful.
"""

import random
from collections import defaultdict

import numpy as np
import torch

from data import DEVICE
import data as _data   # SAE_HOOK is read at runtime via _data.SAE_HOOK
                       # (it's None at import time, set later by load_config())

LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Step 1: Feature Identification
# ---------------------------------------------------------------------------

def get_feature_activations(model, sae, tokens: torch.Tensor) -> torch.Tensor:
    """
    Run one forward pass and return mean SAE feature activations over the sequence.
    Returns: Tensor of shape (d_sae,)
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=_data.SAE_HOOK)
        resid        = cache[_data.SAE_HOOK]     # (batch, seq, d_model)
        feature_acts = sae.encode(resid)         # (batch, seq, d_sae)
        return feature_acts[0].mean(dim=0)       # (d_sae,)


def identify_factual_features(
    model,
    sae,
    tokenizer,
    items: list[dict],
    top_k: int = 50,
) -> tuple[list[int], torch.Tensor]:
    """
    Find SAE features that activate most strongly when the model processes
    the correct answer to a TruthfulQA multiple-choice question.

    Returns:
        factual_features: list of top-k feature indices
        mean_acts:        full mean activation vector (d_sae,) — used for
                          frequency-matched control sampling
    """
    print(f"[experiment] Profiling factual features over {len(items)} samples...")
    accumulator = torch.zeros(sae.cfg.d_sae, device=DEVICE)

    for item in items:
        correct_letter = LETTERS[item["label"]]
        correct_text   = item["choices"][item["label"]]
        prompt = (
            f"Question: {item['question']}\n"
            f"A) {item['choices'][0]}\n"
            f"B) {item['choices'][1]}\n"
            f"C) {item['choices'][2]}\n"
            f"D) {item['choices'][3]}\n"
            f"Answer: {correct_letter}) {correct_text}"
        )
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(DEVICE)

        accumulator += get_feature_activations(model, sae, tokens)

    mean_acts        = accumulator / len(items)
    factual_features = mean_acts.topk(top_k).indices.cpu().tolist()
    print(f"[experiment] Top {top_k} feature indices (first 10): {factual_features[:10]}")
    return factual_features, mean_acts


def sample_random_features(sae, n: int, exclude: list[int]) -> list[int]:
    """
    Sample n uniformly random feature indices, excluding the factual set.
    Basic random control — may include mostly inactive features.
    """
    exclude_set = set(exclude)
    pool = [i for i in range(sae.cfg.d_sae) if i not in exclude_set]
    return random.sample(pool, n)


def sample_frequency_matched_features(
    mean_acts: torch.Tensor,
    factual_features: list[int],
    n: int,
) -> list[int]:
    """
    Sample n features whose mean activation magnitudes are similar to those
    of the factual features, while excluding the factual features themselves.

    Strategy:
      - Compute the mean activation magnitude of the factual feature set
      - Find all non-factual features whose magnitude falls within ±50% of that mean
      - Sample n from that pool uniformly at random

    This is a stronger control than uniform random sampling because it ensures
    the suppressed features are actually active to a similar degree as the
    factual features. Zeroing inactive features has no effect, which would make
    a naive random control trivially similar to baseline.

    Args:
        mean_acts:        full mean activation vector from identify_factual_features()
        factual_features: feature indices to exclude from sampling
        n:                number of features to sample

    Returns:
        List of n frequency-matched feature indices
    """
    factual_set  = set(factual_features)
    factual_mags = mean_acts[factual_features].abs()
    target_mag   = factual_mags.mean().item()
    tolerance    = target_mag * 0.5   # within 50% of target magnitude

    all_mags = mean_acts.abs().cpu()
    candidates = [
        i for i in range(len(mean_acts))
        if i not in factual_set
        and abs(all_mags[i].item() - target_mag) <= tolerance
    ]

    print(f"[experiment] Freq-matched pool: {len(candidates)} candidates "
          f"(target mag={target_mag:.4f} ±{tolerance:.4f})")

    if len(candidates) < n:
        # Widen tolerance if not enough candidates
        print(f"[experiment] Pool too small, widening tolerance to 100%...")
        tolerance = target_mag * 1.0
        candidates = [
            i for i in range(len(mean_acts))
            if i not in factual_set
            and abs(all_mags[i].item() - target_mag) <= tolerance
        ]

    return random.sample(candidates, min(n, len(candidates)))


# ---------------------------------------------------------------------------
# Step 2: Suppression Hook
# ---------------------------------------------------------------------------

def make_suppression_hook(sae, feature_indices: list[int]):
    """
    Build a TransformerLens forward hook that zeroes out the given SAE features.

    At each forward pass:
      1. Encode residual stream → sparse feature activations
      2. Zero out target feature activations
      3. Decode back to residual stream space
      4. Return corrupted residual, replacing the original
    """
    feat_tensor = torch.tensor(feature_indices, device=DEVICE)

    def hook_fn(resid, hook):
        with torch.no_grad():
            feature_acts = sae.encode(resid)
            feature_acts[:, :, feat_tensor] = 0.0
            return sae.decode(feature_acts)

    return hook_fn


# ---------------------------------------------------------------------------
# Step 3: Evaluation — Multiple Choice Log-Prob Scoring
# ---------------------------------------------------------------------------

def _mc_predict(model, tokenizer, item: dict, hook_fn=None) -> int:
    """
    Score each answer choice by the mean log-probability the model assigns to
    the full answer text, given only the question as context (completion scoring).

    For each choice we build "Question: X\nAnswer: <choice text>" and compute
    the mean per-token log-prob over the answer tokens only. The choice with
    the highest score wins.

    This avoids the positional bias problem of letter-token scoring.
    """
    question_prefix = f"Question: {item['question']}\nAnswer:"

    scores = []
    for choice in item["choices"]:
        full_text     = question_prefix + " " + choice
        full_tokens   = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(DEVICE)
        prefix_tokens = tokenizer(
            question_prefix, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(DEVICE)

        n_answer_tokens = full_tokens.shape[1] - prefix_tokens.shape[1]

        if n_answer_tokens <= 0:
            scores.append(-float("inf"))
            continue

        with torch.no_grad():
            if hook_fn is None:
                logits = model(full_tokens)
            else:
                with model.hooks(fwd_hooks=[(_data.SAE_HOOK, hook_fn)]):
                    logits = model(full_tokens)

        log_probs    = torch.log_softmax(logits[0], dim=-1)  # (seq, vocab)
        answer_start = prefix_tokens.shape[1] - 1

        total_lp = 0.0
        for pos in range(answer_start, answer_start + n_answer_tokens):
            target_token = full_tokens[0, pos + 1].item()
            total_lp    += log_probs[pos, target_token].item()

        scores.append(total_lp / n_answer_tokens)

    return int(np.argmax(scores))


def _compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Mean per-token perplexity on a list of texts. Language quality check."""
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(DEVICE)
        with torch.no_grad():
            loss = model(tokens, return_type="loss")
        n = tokens.shape[1] - 1
        total_loss   += loss.item() * n
        total_tokens += n
    return float(np.exp(total_loss / total_tokens))


def run_evaluation(
    model,
    sae,
    tokenizer,
    factual_features: list[int],
    random_features: list[int],
    freq_matched_features: list[int],
    eval_items: list[dict],
    wiki_texts: list[str],
) -> dict:
    """
    Compare four conditions on MC accuracy and perplexity.

    Conditions:
        baseline       — no intervention
        targeted       — factual SAE features suppressed
        random ctrl    — equal number of uniformly random features suppressed
        freq-matched   — features matched by activation frequency suppressed

    Returns dict with accuracy for each condition + perplexity.
    """
    print(f"\n[experiment] Evaluating {len(eval_items)} TruthfulQA MC samples...")
    print(f"[experiment] Four conditions: baseline / targeted / random / freq-matched")

    targeted_hook  = make_suppression_hook(sae, factual_features)
    random_hook    = make_suppression_hook(sae, random_features)
    freq_hook      = make_suppression_hook(sae, freq_matched_features)

    counts = defaultdict(int)

    for i, item in enumerate(eval_items):
        correct = item["label"]

        base_pred  = _mc_predict(model, tokenizer, item)
        tgt_pred   = _mc_predict(model, tokenizer, item, hook_fn=targeted_hook)
        rand_pred  = _mc_predict(model, tokenizer, item, hook_fn=random_hook)
        freq_pred  = _mc_predict(model, tokenizer, item, hook_fn=freq_hook)

        counts["baseline"] += int(base_pred == correct)
        counts["targeted"] += int(tgt_pred  == correct)
        counts["random"]   += int(rand_pred == correct)
        counts["freq"]     += int(freq_pred == correct)

        if (i + 1) % 50 == 0:
            n = i + 1
            print(f"  [{n}/{len(eval_items)}]  "
                  f"base={counts['baseline']/n:.2f}  "
                  f"targeted={counts['targeted']/n:.2f}  "
                  f"random={counts['random']/n:.2f}  "
                  f"freq={counts['freq']/n:.2f}")
            print(f"    Q: {item['question'][:70]}")
            print(f"    Correct: {LETTERS[correct]})  "
                  f"Base: {LETTERS[base_pred]})  "
                  f"Targeted: {LETTERS[tgt_pred]})")

    n = len(eval_items)
    print("\n[experiment] Computing perplexity on WikiText-103...")
    ppl = _compute_perplexity(model, tokenizer, wiki_texts)

    return {
        "n_eval":                    n,
        "chance_accuracy":           0.25,
        "baseline_accuracy":         counts["baseline"] / n,
        "targeted_accuracy":         counts["targeted"] / n,
        "random_control_accuracy":   counts["random"]   / n,
        "freq_matched_accuracy":     counts["freq"]     / n,
        "baseline_ppl":              ppl,
    }