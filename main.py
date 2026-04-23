"""
main.py — Entry point
======================
Wires data.py and experiment.py together. Nothing scientific lives here.

Usage:
    # Step 1: download datasets once
    python download_data.py

    # Step 2: smoke test (CPU-safe, confirms the pipeline runs end-to-end)
    python main.py --smoke-test

    # Step 3: standard run
    python main.py

    # Larger run (needs GPU)
    python main.py --n-profile 500 --n-eval 200 --top-k 100
"""

import sys
print("Python started, loading imports...", flush=True)

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "info"   # shows download progress before anything else

print("Importing torch (this can take 1-2 min on first run)...", flush=True)
import torch
print("torch loaded. Importing transformers...", flush=True)
import json
from pathlib import Path
print("All imports done.\n", flush=True)

import argparse
import random


from data import load_model_and_sae, load_trivia_train, load_trivia_val, load_wikitext
from experiment import (
    identify_factual_features,
    sample_random_features,
    run_evaluation,
)

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="SAE Factual Amnesia Experiment")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Tiny run (5 samples) to verify the pipeline works.")
    p.add_argument("--n-profile",   type=int, default=200,
                   help="TriviaQA samples used to identify factual features.")
    p.add_argument("--n-eval",      type=int, default=100,
                   help="TriviaQA samples used for evaluation.")
    p.add_argument("--top-k",       type=int, default=50,
                   help="Number of SAE features to flag as factual.")
    p.add_argument("--output",      type=str, default="results.json",
                   help="Where to save the JSON results.")
    return p.parse_args()


def print_results(results: dict):
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"  Eval samples:        {results['n_eval']}")
    print(f"  Baseline accuracy:   {results['baseline_accuracy']:.3f}")
    print(f"  Targeted accuracy:   {results['targeted_accuracy']:.3f}  <- want this lower")
    print(f"  Control accuracy:    {results['control_accuracy']:.3f}  <- want this near baseline")
    print(f"  Baseline perplexity: {results['baseline_ppl']:.2f}  <- language quality check")
    print("=" * 55)

    delta_targeted = results["baseline_accuracy"] - results["targeted_accuracy"]
    delta_control  = results["baseline_accuracy"] - results["control_accuracy"]
    print(f"\n  Targeted drop: {delta_targeted:.3f}")
    print(f"  Control drop:  {delta_control:.3f}")

    if delta_targeted > delta_control + 0.05:
        print("\n  [PASS] Targeted corruption outperforms random control.")
        print("         Factual features are doing real work.")
    else:
        print("\n  [FAIL] Targeted and control drops are similar.")
        print("         Try adjusting --top-k or the SAE layer in data.py.")


def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.smoke_test:
        print("[main] Smoke test mode -- using minimal sample counts.")
        args.n_profile = 5
        args.n_eval    = 5
        args.top_k     = 10

    # Load model and SAE (HuggingFace caches weights after first download)
    model, sae, tokenizer = load_model_and_sae()

    # Load datasets from local files (written by download_data.py)
    profile_items = load_trivia_train(n=args.n_profile)
    eval_items    = load_trivia_val(n=args.n_eval)
    wiki_texts    = load_wikitext(n=50)

    # Identify factual features
    factual_features = identify_factual_features(
        model, sae, tokenizer,
        trivia_items=profile_items,
        top_k=args.top_k,
    )

    # Sample random control features
    random_features = sample_random_features(
        sae, n=len(factual_features), exclude=factual_features
    )

    # Run evaluation
    results = run_evaluation(
        model, sae, tokenizer,
        factual_features=factual_features,
        random_features=random_features,
        eval_items=eval_items,
        wiki_texts=wiki_texts,
    )

    # Print and save
    print_results(results)

    out = {**results, "factual_features": factual_features, "random_features": random_features}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[main] Results saved to {args.output}")


if __name__ == "__main__":
    main()