"""
experiment.py — Core experiment logic
======================================
Three steps:
  1. identify_factual_features  — find which SAE features activate on correct answers
  2. make_suppression_hook      — build a hook that zeroes those features at inference time
  3. run_evaluation             — compare baseline vs. targeted vs. random-control

This is the v1 (basic) implementation. Only suppression is implemented here.
Noise injection and feature swapping can be added later as separate hook factories.
"""

import random
from collections import defaultdict

import numpy as np
import torch

from data import SAE_HOOK, DEVICE


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
        resid = cache[SAE_HOOK]               # (batch, seq, d_model)
        feature_acts = sae.encode(resid)      # (batch, seq, d_sae)
        return feature_acts[0].mean(dim=0)    # (d_sae,)  — average over seq positions


def identify_factual_features(
    model,
    sae,
    tokenizer,
    trivia_items: list[dict],
    top_k: int = 50,
) -> list[int]:
    """
    Find the SAE features that activate most strongly on correct TriviaQA answers.

    For each QA pair we pass the full "Question: X\\nAnswer: Y" string through
    the model and record the mean feature activation vector. We accumulate a
    running sum across all items, then return the top-k feature indices.

    Args:
        trivia_items: list of TriviaQA items (from data.load_trivia_qa)
        top_k:        how many features to flag as "factual"

    Returns:
        List of integer feature indices, length == top_k
    """
    print(f"[experiment] Profiling factual features over {len(trivia_items)} samples...")
    accumulator = torch.zeros(sae.cfg.d_sae, device=DEVICE)

    for item in trivia_items:
        prompt = f"Question: {item['question']}\nAnswer: {item['answer']['value']}"
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.to(DEVICE)

        accumulator += get_feature_activations(model, sae, tokens)

    mean_acts = accumulator / len(trivia_items)
    factual_features = mean_acts.topk(top_k).indices.cpu().tolist()
    print(f"[experiment] Top {top_k} feature indices (first 10): {factual_features[:10]}")
    return factual_features


def sample_random_features(sae, n: int, exclude: list[int]) -> list[int]:
    """
    Sample n random feature indices, excluding the factual set.
    Used for the control condition — suppressing random features instead of factual ones.
    """
    exclude_set = set(exclude)
    pool = [i for i in range(sae.cfg.d_sae) if i not in exclude_set]
    return random.sample(pool, n)


# ---------------------------------------------------------------------------
# Step 2: Suppression Hook (v1 intervention)
# ---------------------------------------------------------------------------

def make_suppression_hook(sae, feature_indices: list[int]):
    """
    Build a TransformerLens forward hook that suppresses the given SAE features.

    At each forward pass the hook:
      1. Encodes the residual stream through the SAE  → sparse feature activations
      2. Zeroes out the target feature activations
      3. Decodes back to residual-stream space
      4. Returns the corrupted residual, replacing the original

    Args:
        feature_indices: list of SAE feature indices to suppress

    Returns:
        A hook function compatible with TransformerLens model.hooks()
    """
    feat_tensor = torch.tensor(feature_indices, device=DEVICE)

    def hook_fn(resid, hook):
        # resid shape: (batch, seq, d_model)
        with torch.no_grad():
            feature_acts = sae.encode(resid)           # (batch, seq, d_sae)
            feature_acts[:, :, feat_tensor] = 0.0      # zero out target features
            return sae.decode(feature_acts)             # (batch, seq, d_model)

    return hook_fn


# ---------------------------------------------------------------------------
# Step 3: Evaluation
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, hook_fn=None, max_new_tokens: int = 40) -> str:
    """Generate a completion, optionally applying a residual-stream hook."""
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=128
    ).input_ids.to(DEVICE)

    with torch.no_grad():
        if hook_fn is None:
            output = model.generate(tokens, max_new_tokens=max_new_tokens)
        else:
            with model.hooks(fwd_hooks=[(SAE_HOOK, hook_fn)]):
                output = model.generate(tokens, max_new_tokens=max_new_tokens)

    generated_ids = output[0, tokens.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _exact_match(prediction: str, aliases: list[str]) -> bool:
    """Normalised exact match: does the prediction start with any valid answer?"""
    pred = prediction.strip().lower()
    return any(pred.startswith(a.strip().lower()) for a in aliases)


def _compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Mean per-token perplexity of the *base* model on a list of texts."""
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
    Compare three conditions on factual accuracy and perplexity.

    Conditions:
        baseline — no intervention
        targeted — factual SAE features suppressed
        control  — equal number of random features suppressed

    Args:
        factual_features: output of identify_factual_features()
        random_features:  output of sample_random_features()
        eval_items:       TriviaQA validation items (from data.load_trivia_qa)
        wiki_texts:       WikiText paragraphs for perplexity (from data.load_wikitext)

    Returns:
        dict with accuracy and perplexity for each condition
    """
    print(f"\n[experiment] Evaluating {len(eval_items)} TriviaQA samples...")

    targeted_hook = make_suppression_hook(sae, factual_features)
    control_hook  = make_suppression_hook(sae, random_features)

    counts = defaultdict(int)

    for i, item in enumerate(eval_items):
        aliases = item["answer"]["aliases"]
        prompt  = f"Question: {item['question']}\nAnswer:"

        base_ans     = _generate(model, tokenizer, prompt)
        targeted_ans = _generate(model, tokenizer, prompt, hook_fn=targeted_hook)
        control_ans  = _generate(model, tokenizer, prompt, hook_fn=control_hook)

        counts["baseline"] += int(_exact_match(base_ans, aliases))
        counts["targeted"] += int(_exact_match(targeted_ans, aliases))
        counts["control"]  += int(_exact_match(control_ans, aliases))

        # Print a progress update + a sample output every 10 items
        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"  [{n}/{len(eval_items)}]  "
                  f"base={counts['baseline']/n:.2f}  "
                  f"targeted={counts['targeted']/n:.2f}  "
                  f"control={counts['control']/n:.2f}")
            print(f"    Q: {item['question'][:60]}")
            print(f"    baseline → {base_ans[:60]}")
            print(f"    targeted → {targeted_ans[:60]}")

    n = len(eval_items)
    print("\n[experiment] Computing perplexity on WikiText-103...")
    ppl = _compute_perplexity(model, tokenizer, wiki_texts)

    return {
        "n_eval":             n,
        "baseline_accuracy":  counts["baseline"] / n,
        "targeted_accuracy":  counts["targeted"] / n,
        "control_accuracy":   counts["control"]  / n,
        "baseline_ppl":       ppl,
    }
