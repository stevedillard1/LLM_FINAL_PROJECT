# Inducing Controlled Factual Amnesia in LLMs

**Steven Dillard** — sole contributor (model setup, data pipeline, experiment design, evaluation, writeup)

---

## What This Is

This project tests whether Sparse Autoencoders (SAEs) can selectively corrupt factual knowledge in a language model while preserving general language fluency — a kind of simulated semantic amnesia. See [`WRITEUP.md`](WRITEUP.md) for the full project writeup.

---

## Repo Map

```
.
├── sae-factual-amnesia
    ├── main.py              # Entry point — wires everything together, CLI args
    ├── experiment.py        # Core logic: feature identification, hooks, evaluation
    ├── data.py              # All I/O: model/SAE loading, dataset loading from disk
    ├── download_data.py     # Run once to cache datasets locally
├── results.json         # Output from the last experiment run
├── pyproject.toml     # Python dependencies
├── WRITEUP.md           # Full project writeup
└── README.md            # This file
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
uv sync
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Download and cache datasets (run once)
uv run python download_data.py

# 3. Smoke test — confirms the pipeline runs end to end
uv run python main.py --smoke-test
```

> **Note:** The model (`EleutherAI/pythia-70m-deduped`, ~150MB) is downloaded automatically by HuggingFace on first run and cached in `~/.cache/huggingface`. You do not need to download it manually. Set `HF_TOKEN` in your environment to avoid rate limiting.

---

## Running the Experiment

```bash
# Standard run (200 profiling samples, 100 eval samples, suppression intervention)
uv run python main.py

# Adjust sample counts
uv run python main.py --n-profile 500 --n-eval 200 --top-k 100

# Save results to a specific file
uv run python main.py --output results_run1.json
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--smoke-test` | off | Tiny 5-sample run to verify the pipeline |
| `--n-profile` | 200 | TriviaQA samples used to identify factual features |
| `--n-eval` | 100 | TriviaQA samples used for evaluation |
| `--top-k` | 50 | Number of SAE features flagged as factual |
| `--output` | `results.json` | Path to save JSON results |

---

## Results

Results are saved to `results.json` after each run. Key fields:

| Field | What it means |
|---|---|
| `baseline_accuracy` | Model accuracy with no intervention |
| `targeted_accuracy` | Accuracy after suppressing factual features |
| `control_accuracy` | Accuracy after suppressing random features |
| `baseline_ppl` | WikiText-103 perplexity (language quality check) |

**Interpreting the output:**
- `targeted_accuracy << control_accuracy` → feature identification is working
- `targeted_accuracy ≈ control_accuracy` → try adjusting `--top-k` or the SAE layer in `data.py`
- Perplexity spikes alongside accuracy → factual and fluency features may be entangled

---

## Current Model / SAE

Configured in `data.py`:

```python
MODEL_ID    = "EleutherAI/pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm"
SAE_HOOK    = "blocks.3.hook_resid_post"
```

To experiment with a different layer, change `SAE_HOOK`. Valid hook IDs for this release:
`blocks.0–5.hook_resid_pre` and `blocks.0–5.hook_resid_post`.
