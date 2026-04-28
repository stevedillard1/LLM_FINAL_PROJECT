"""
main.py — Entry point
======================
Wires data.py and experiment.py together.

Usage:
    # Step 1: download datasets once
    python download_data.py

    # Step 2: smoke test
    python main.py --smoke-test

    # Step 3: full run
    python main.py
"""

import argparse
import json
import random
from pathlib import Path

import torch

from data import load_model_and_sae, load_truthful_qa, load_wikitext
from experiment import identify_factual_features, sample_random_features, run_evaluation

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="SAE Factual Amnesia Experiment")
    p.add_argument("--smoke-test", action="store_true",
                   help="5-sample run to verify the pipeline works.")
    p.add_argument("--n-profile",  type=int, default=200,
                   help="TruthfulQA samples used to identify factual features.")
    p.add_argument("--n-eval",     type=int, default=100,
                   help="TruthfulQA samples used for evaluation.")
    p.add_argument("--top-k",      type=int, default=50,
                   help="Number of SAE features to flag as factual.")
    p.add_argument("--output",     type=str, default="results.json",
                   help="Where to save JSON results.")
    return p.parse_args()


def print_results(results: dict):
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"  Eval samples:        {results['n_eval']}")
    print(f"  Chance accuracy:     {results['chance_accuracy']:.2f}  (4-way random baseline)")
    print(f"  Baseline accuracy:   {results['baseline_accuracy']:.3f}")
    print(f"  Targeted accuracy:   {results['targeted_accuracy']:.3f}  <- want this lower")
    print(f"  Control accuracy:    {results['control_accuracy']:.3f}  <- want this near baseline")
    print(f"  Baseline perplexity: {results['baseline_ppl']:.2f}")
    print("=" * 55)

    delta_t = results["baseline_accuracy"] - results["targeted_accuracy"]
    delta_c = results["baseline_accuracy"] - results["control_accuracy"]
    print(f"\n  Targeted drop: {delta_t:.3f}")
    print(f"  Control drop:  {delta_c:.3f}")

    if results["baseline_accuracy"] <= results["chance_accuracy"] + 0.05:
        print("\n  [WARN] Baseline is near chance — model may not be suitable.")
        print("         Consider switching to a larger or instruction-tuned model.")
    elif delta_t > delta_c + 0.05:
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
        print("[main] Smoke test mode -- minimal sample counts.")
        args.n_profile = 5
        args.n_eval    = 5
        args.top_k     = 10

    # Load
    model, sae, tokenizer = load_model_and_sae()
    all_items  = load_truthful_qa(n=args.n_profile + args.n_eval)
    wiki_texts = load_wikitext(n=50)

    # Split profile vs eval to avoid overlap
    profile_items = all_items[:args.n_profile]
    eval_items    = all_items[args.n_profile:args.n_profile + args.n_eval]

    # Identify factual features
    factual_features = identify_factual_features(
        model, sae, tokenizer,
        items=profile_items,
        top_k=args.top_k,
    )

    # Sample random control features
    random_features = sample_random_features(
        sae, n=len(factual_features), exclude=factual_features
    )

    # Evaluate
    results = run_evaluation(
        model, sae, tokenizer,
        factual_features=factual_features,
        random_features=random_features,
        eval_items=eval_items,
        wiki_texts=wiki_texts,
    )

    print_results(results)

    out = {**results, "factual_features": factual_features, "random_features": random_features}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[main] Results saved to {args.output}")


if __name__ == "__main__":
    main()