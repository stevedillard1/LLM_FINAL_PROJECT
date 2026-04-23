# Inducing Controlled Factual Amnesia in LLMs
## SAE-based Feature Corruption Experiment

---

### Overview

This project tests whether Sparse Autoencoders (SAEs) can be used to selectively corrupt
factual knowledge in Pythia-1.4B while preserving general language capabilities — a form
of simulated semantic amnesia.

**Three experimental conditions:**
| Condition | Description |
|---|---|
| Baseline | Unmodified Pythia-1.4B |
| Targeted | Top-k factual SAE features suppressed/corrupted |
| Control | Equal number of random non-factual features suppressed |

The control condition is critical: if targeted and random conditions perform similarly,
it suggests feature identification needs work, not that the idea is wrong.

---

### Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Smoke test (CPU-safe, ~2 minutes, confirms the pipeline runs)
python experiment.py --smoke-test
```

---

### Running the Full Experiment

```bash
# Default: suppression intervention, 200 profiling samples, 100 eval samples
python experiment.py

# Try noise injection instead of suppression
python experiment.py --intervention noise

# Try activation swapping (most aggressive)
python experiment.py --intervention swap

# Full run with more samples (needs GPU)
python experiment.py \
  --n-factual-samples 500 \
  --n-eval-samples 200 \
  --top-k-features 100 \
  --intervention suppression \
  --output results_suppression.json
```

**Tip:** Run all three interventions and compare results:
```bash
for mode in suppression noise swap; do
  python experiment.py --intervention $mode --output results_${mode}.json
done
```

---

### How It Works

#### 1. Feature Identification (`identify_factual_features`)
- Runs Pythia-1.4B on TriviaQA QA pairs formatted as `"Question: X\nAnswer: Y"`
- Captures SAE feature activations at layer 12 (`blocks.12.hook_resid_post`)
- Averages activations across all samples
- Returns the top-k most consistently activated features = "factual features"

#### 2. Intervention Hooks
Three strategies are implemented via TransformerLens forward hooks:

**Suppression** — zeros out target feature activations:
```
encode(resid) → zero out features[i] → decode → replace resid
```

**Noise injection** — adds scaled Gaussian noise to target features:
```
encode(resid) → add noise to features[i] → decode → replace resid
```

**Swap** — replaces factual feature activations with unrelated feature activations:
```
encode(resid) → features[factual] = features[unrelated] → decode → replace resid
```

#### 3. Evaluation
- **Factual accuracy**: TriviaQA exact match (normalised, checks all valid aliases)
- **Perplexity**: Measured on WikiText-103 (language quality check)
- **BERTScore**: Targeted outputs vs. reference answers + vs. baseline outputs

---

### Reading the Results

```
Baseline accuracy:      0.35   ← what the model can do normally
Targeted accuracy:      0.10   ← should drop significantly
Control accuracy:       0.32   ← should stay near baseline

BERTScore (tgt vs ref): 0.42   ← how semantically close targeted outputs are to correct answers
BERTScore (tgt vs base):0.71   ← how fluent/similar targeted outputs are to baseline (want high)
```

**The key diagnostic:**
- `targeted_accuracy << control_accuracy` → your feature identification worked
- `targeted_accuracy ≈ control_accuracy` → you corrupted the wrong features; try adjusting
  `--top-k-features` or profiling on a different layer (edit `SAE_ID` in experiment.py)
- Perplexity spikes as much as accuracy drops → factual/fluency features are entangled;
  consider targeting a different SAE layer or switching to weight editing

---

### Extending the Project

**Try a different SAE layer:**
Edit `SAE_ID` in `experiment.py`. Earlier layers (e.g., `blocks.6`) encode more syntactic
features; later layers (e.g., `blocks.18`) are more semantic. Layer 12 is a reasonable
starting point for factual content.

**Add a linear probe for validation:**
Use Representation Engineering (Zou et al., 2023) to train a true/false probe on the
residual stream, then check that your identified factual features correlate with the
probe's decision boundary.

**Weight editing (more permanent):**
Instead of activation hooks, modify `sae.W_dec` directly to make factual features decode
to zero vectors. This makes the corruption persistent across all forward passes without hooks.

---

### Citation

If you use this code, please cite:
- Bricken et al., "Towards Monosemanticity", Transformer Circuits Thread, 2023
- Templeton et al., "Scaling Monosemanticity", Transformer Circuits Thread, 2024
- Zou et al., "Representation Engineering", arXiv 2023
