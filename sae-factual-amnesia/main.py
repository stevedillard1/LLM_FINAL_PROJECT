"""
main.py — Entry point
======================
All configuration lives in config.yaml — edit that file to change the model
or experiment parameters. No code changes needed.

Usage:
    python download_data.py     # run once to cache datasets
    python main.py              # run experiment using config.yaml
    python main.py --smoke-test # 5-sample test to verify the pipeline
"""

import argparse
import json
import random
from pathlib import Path

import torch

from data import load_config, load_model_and_sae, load_truthful_qa, load_wikitext
from experiment import (
    identify_factual_features,
    sample_random_features,
    sample_frequency_matched_features,
    run_evaluation,
)

SEED             = 42
TRUTHFULQA_TOTAL = 817


def parse_args():
    p = argparse.ArgumentParser(description="SAE Factual Amnesia Experiment")
    p.add_argument("--smoke-test",  action="store_true",
                   help="5-sample run to verify the pipeline works.")
    p.add_argument("--config",      type=str, default="config.yaml",
                   help="Path to config file (default: config.yaml).")
    return p.parse_args()


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Eval samples:             {results['n_eval']}")
    print(f"  Chance accuracy:          {results['chance_accuracy']:.2f}  (4-way random)")
    print(f"  Baseline accuracy:        {results['baseline_accuracy']:.3f}")
    print(f"  Targeted accuracy:        {results['targeted_accuracy']:.3f}  <- want lower")
    print(f"  Random control:           {results['random_control_accuracy']:.3f}  <- want near baseline")
    print(f"  Freq-matched control:     {results['freq_matched_accuracy']:.3f}  <- stronger control")
    print(f"  Baseline perplexity:      {results['baseline_ppl']:.2f}")
    print("=" * 60)

    dt = results["baseline_accuracy"] - results["targeted_accuracy"]
    dr = results["baseline_accuracy"] - results["random_control_accuracy"]
    df = results["baseline_accuracy"] - results["freq_matched_accuracy"]
    print(f"\n  Targeted drop:      {dt:.3f}")
    print(f"  Random drop:        {dr:.3f}")
    print(f"  Freq-matched drop:  {df:.3f}")

    if results["baseline_accuracy"] <= results["chance_accuracy"] + 0.05:
        print("\n  [WARN] Baseline near chance — model may not suit this eval.")
    elif dt > df + 0.05:
        print("\n  [PASS] Targeted drop exceeds freq-matched control.")
        print("         Factual features are doing meaningful work.")
    elif dt > dr + 0.05:
        print("\n  [PARTIAL] Targeted beats random but not freq-matched control.")
        print("            Activation frequency may partially explain the drop.")
    else:
        print("\n  [FAIL] All conditions similar.")
        print("         Try increasing top_k in config.yaml or a different layer.")


def main():
    args  = parse_args()
    cfg   = load_config(args.config)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Read parameters from config (smoke-test overrides everything)
    if args.smoke_test:
        print("[main] Smoke test mode — overriding config with minimal counts.")
        n_profile  = 5
        n_eval     = 5
        top_k      = 10
        n_wikitext = 10
    else:
        n_profile  = cfg.get("n_profile",  400)
        n_eval     = cfg.get("n_eval",     417)
        top_k      = cfg.get("top_k",      100)
        n_wikitext = cfg.get("n_wikitext", 100)

    output = cfg.get("output", "results.json")

    if n_profile + n_eval > TRUTHFULQA_TOTAL:
        print(f"[main] WARNING: n_profile + n_eval ({n_profile + n_eval}) "
              f"exceeds TruthfulQA total ({TRUTHFULQA_TOTAL}). Capping n_eval.")
        n_eval = TRUTHFULQA_TOTAL - n_profile

    # Load model and datasets
    model, sae, tokenizer = load_model_and_sae()
    all_items  = load_truthful_qa(n=n_profile + n_eval)
    wiki_texts = load_wikitext(n=n_wikitext)

    profile_items = all_items[:n_profile]
    eval_items    = all_items[n_profile:n_profile + n_eval]

    # Feature identification
    factual_features, mean_acts = identify_factual_features(
        model, sae, tokenizer,
        items=profile_items,
        top_k=top_k,
    )

    # Control feature sets
    random_features = sample_random_features(
        sae, n=len(factual_features), exclude=factual_features
    )
    freq_matched_features = sample_frequency_matched_features(
        mean_acts, factual_features, n=len(factual_features)
    )

    print(f"[main] Factual features (first 5):     {factual_features[:5]}")
    print(f"[main] Random control (first 5):       {random_features[:5]}")
    print(f"[main] Freq-matched control (first 5): {freq_matched_features[:5]}")

    # Run all four conditions
    results = run_evaluation(
        model, sae, tokenizer,
        factual_features=factual_features,
        random_features=random_features,
        freq_matched_features=freq_matched_features,
        eval_items=eval_items,
        wiki_texts=wiki_texts,
    )

    print_results(results)

    out = {
        **results,
        "factual_features":      factual_features,
        "random_features":       random_features,
        "freq_matched_features": freq_matched_features,
        "config": {
            "model_preset": cfg.get("model_preset"),
            "n_profile":    n_profile,
            "n_eval":       n_eval,
            "top_k":        top_k,
        }
    }
    Path(output).write_text(json.dumps(out, indent=2))
    print(f"\n[main] Results saved to {output}")


if __name__ == "__main__":
    main()