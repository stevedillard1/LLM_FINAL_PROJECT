"""
experiment.py — Core experiment logic
======================================
Three steps:
  1. identify_factual_features  — find SAE features that activate on correct MC answers
  2. make_suppression_hook      — zero out those features at inference time
  3. run_evaluation             — compare baseline vs. targeted vs. random-control

Evaluation uses multiple-choice scoring (log-prob of each choice token) instead
of free-text generation, so even smaller models produce measurable baseline accuracy.
"""

import random
from collections import defaultdict

import numpy as np
import torch

from data import SAE_HOOK, DEVICE

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
        _, cache = model.run_with_cache(tokens, names_filter=SAE_HOOK)
        resid        = cache[SAE_HOOK]           # (batch, seq, d_model)
        feature_acts = sae.encode(resid)         # (batch, seq, d_sae)
        return feature_acts[0].mean(dim=0)       # (d_sae,)


def identify_factual_features(
    model,
    sae,
    tokenizer,
    items: list[dict],
    top_k: int = 50,
) -> list[int]:
    """
    Find SAE features that activate most strongly when the model processes
    the correct answer to a TruthfulQA multiple-choice question.

    For each item we build a prompt that includes the question + correct answer
    and record mean SAE activations. We accumulate across all items and return
    the top-k most consistently activated feature indices.

    Args:
        items:  TruthfulQA MC items (from data.load_truthful_qa)
        top_k:  how many features to flag as factual

    Returns:
        List of integer feature indices, length == top_k
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
    return factual_features


def sample_random_features(sae, n: int, exclude: list[int]) -> list[int]:
    """
    Sample n random feature indices, excluding the factual set.
    Used for the control condition.
    """
    exclude_set = set(exclude)
    pool = [i for i in range(sae.cfg.d_sae) if i not in exclude_set]
    return random.sample(pool, n)


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

    Args:
        feature_indices: SAE feature indices to suppress

    Returns:
        Hook function compatible with TransformerLens model.hooks()
    """
    feat_tensor = torch.tensor(feature_indices, device=DEVICE)

    def hook_fn(resid, hook):
        # resid: (batch, seq, d_model)
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
    the full answer text, given only the question as context.

    This is completion scoring: for each choice we build
        "Question: X\nAnswer: <choice text>"
    and compute the mean per-token log-prob over the answer tokens only.
    The choice with the highest score wins.

    This avoids the positional bias problem of letter-token scoring (where small
    base models default to always picking A) by using the actual answer text as
    the scoring signal instead of a single letter token.
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

        # Need at least one answer token to score
        if n_answer_tokens <= 0:
            scores.append(-float("inf"))
            continue

        with torch.no_grad():
            if hook_fn is None:
                logits = model(full_tokens)
            else:
                with model.hooks(fwd_hooks=[(SAE_HOOK, hook_fn)]):
                    logits = model(full_tokens)

        # Shift: logits[i] predicts token[i+1]
        # We want log-probs over the answer tokens only
        log_probs   = torch.log_softmax(logits[0], dim=-1)  # (seq, vocab)
        answer_start = prefix_tokens.shape[1] - 1           # position before first answer token

        total_lp = 0.0
        for pos in range(answer_start, answer_start + n_answer_tokens):
            target_token = full_tokens[0, pos + 1].item()
            total_lp    += log_probs[pos, target_token].item()

        scores.append(total_lp / n_answer_tokens)           # mean per-token log-prob

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
    eval_items: list[dict],
    wiki_texts: list[str],
) -> dict:
    """
    Compare three conditions on MC accuracy and perplexity.

    Conditions:
        baseline — no intervention
        targeted — factual SAE features suppressed
        control  — equal number of random features suppressed

    Returns:
        dict with accuracy and perplexity for each condition
    """
    print(f"\n[experiment] Evaluating {len(eval_items)} TruthfulQA MC samples...")

    targeted_hook = make_suppression_hook(sae, factual_features)
    control_hook  = make_suppression_hook(sae, random_features)

    counts = defaultdict(int)

    for i, item in enumerate(eval_items):
        correct = item["label"]

        base_pred     = _mc_predict(model, tokenizer, item)
        targeted_pred = _mc_predict(model, tokenizer, item, hook_fn=targeted_hook)
        control_pred  = _mc_predict(model, tokenizer, item, hook_fn=control_hook)

        counts["baseline"] += int(base_pred     == correct)
        counts["targeted"] += int(targeted_pred == correct)
        counts["control"]  += int(control_pred  == correct)

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"  [{n}/{len(eval_items)}]  "
                  f"base={counts['baseline']/n:.2f}  "
                  f"targeted={counts['targeted']/n:.2f}  "
                  f"control={counts['control']/n:.2f}")
            print(f"    Q: {item['question'][:70]}")
            print(f"    Correct: {LETTERS[correct]})  "
                  f"Base: {LETTERS[base_pred]})  "
                  f"Targeted: {LETTERS[targeted_pred]})")

    n = len(eval_items)
    print("\n[experiment] Computing perplexity on WikiText-103...")
    ppl = _compute_perplexity(model, tokenizer, wiki_texts)

    return {
        "n_eval":            n,
        "baseline_accuracy": counts["baseline"] / n,
        "targeted_accuracy": counts["targeted"] / n,
        "control_accuracy":  counts["control"]  / n,
        "baseline_ppl":      ppl,
        "chance_accuracy":   0.25,   # 4-way MC random baseline
    }