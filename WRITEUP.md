# Inducing Controlled Factual Amnesia in Large Language Models
### Using Sparse Autoencoders to Selectively Corrupt Factual Feature Representations

**Steven Dillard**
LLM Interpretability — Final Project

---

## 1. Research Question and Motivation

Most interpretability research on large language models asks how to make models more
reliable and less prone to hallucination. This project inverts that question entirely:
can we use Sparse Autoencoders (SAEs) to *deliberately* sever a model's connection to
factual knowledge while keeping its language capabilities intact?

The practical motivation is not to build a worse model, but to use targeted corruption
as a diagnostic tool. If we can identify and suppress the features responsible for
factual recall, it provides strong evidence that those features are doing real
interpretable work in the model's representation space. If we cannot, that is equally
informative — it suggests factual knowledge is not cleanly localized and may be more
distributed than current SAE research implies.

The framing throughout this project is "semantic amnesia": a model that speaks fluently
but has lost the ability to recall ground truth, analogous to a person who retains
language and reasoning capabilities following memory loss.

---

## 2. Related Work

**Towards Monosemanticity (Bricken et al., Anthropic 2023)** is the foundational work
this project builds on. It demonstrated that sparse autoencoders applied to transformer
residual streams can decompose superimposed, polysemantic feature representations into
interpretable, approximately monosemantic components. This is the core tool used to
identify and target factual features.

**Scaling Monosemanticity (Templeton et al., Anthropic 2024)** is the closest
conceptual relative. Anthropic showed that clamping a single SAE feature — the "Golden
Gate Bridge" feature in Claude 3 Sonnet — to a high constant value persistently biased
the model's behavior across all outputs. This project is the adversarial inverse:
rather than amplifying one feature to steer the model, we suppress an entire class of
features to degrade factual recall.

**Representation Engineering (Zou et al., 2023)** identified "truth directions" in
residual stream activations using linear probes trained on true/false statement pairs.
This is directly relevant as an independent method for validating that the SAE features
we identify actually correspond to factual encoding rather than correlated syntactic or
domain features.

**ROME and MEMIT (Meng et al., 2022/2023)** demonstrated via causal tracing that
factual associations in GPT-style models are largely localized to specific mid-layer
MLP computations, and can be precisely edited. This project can be thought of as
anti-MEMIT: rather than inserting factual memories, we are attempting to erase them.
ROME's causal tracing methodology also informs which layers to target.

**Hallucination reduction work** (RLFT, Goodfire AI 2025; various RAG and calibration
approaches) all reinforce factual grounding. This project occupies the underexplored
opposite end: studying what factual representations look like by seeing what happens
when they are deliberately removed.

---

## 3. Method

### 3.1 Model and SAE

Experiments use `EleutherAI/pythia-70m-deduped` loaded via TransformerLens, with a
pretrained SAE from the SAELens library (`pythia-70m-deduped-res-sm`, residual stream,
`blocks.3.hook_resid_post`). Pythia-70m was chosen for memory constraints on the
development machine; the experiment design is model-agnostic and can be run on
Pythia-1.4B on a GPU.

### 3.2 Feature Identification

To identify factual features, we run the model on 200 TriviaQA training examples
formatted as `"Question: X\nAnswer: Y"`. For each example we capture SAE feature
activations at the hook layer, averaged over sequence positions. We then accumulate a
mean activation vector across all examples and return the top-k most consistently
activated feature indices as our "factual feature set."

This is a deliberately simple approach for v1. The key assumption is that features
which activate most strongly and consistently on factual QA content are likely to be
involved in factual recall. A more rigorous approach would use a contrastive probe
(true vs. false statements, following Zou et al.) to identify features that specifically
encode factual *correctness* rather than just factual *content*.

### 3.3 Interventions

We implement one intervention for the baseline experiment: **suppression**, which
zeroes out the target features at inference time via a TransformerLens forward hook:

```
encode(resid) → zero out features[factual_indices] → decode → replace resid
```

The hook intercepts the residual stream at `blocks.3.hook_resid_post`, encodes it
through the SAE to obtain sparse feature activations, zeroes out the targeted features,
and decodes back to the residual stream space before continuing the forward pass.

### 3.4 Experimental Conditions

Three conditions are compared on every evaluation sample:

| Condition | Description |
|---|---|
| Baseline | Unmodified Pythia-70m, no hook |
| Targeted | Top-k factual SAE features suppressed |
| Control | Equal number of randomly sampled non-factual features suppressed |

The control condition is critical. Without it, any accuracy drop could be attributed to
general model degradation from residual stream interference rather than principled
feature suppression. A targeted drop that significantly exceeds the control drop is
evidence that the identified features are doing real factual work.

### 3.5 Evaluation

- **Factual accuracy**: TriviaQA exact match on 100 validation examples, normalized
  (strip, lowercase, check against all valid answer aliases)
- **Perplexity**: Mean per-token perplexity on 50 WikiText-103 paragraphs, measured on
  the unmodified model as a language quality baseline
- Results saved to `results.json`

---

## 4. Experiments and Results

### 4.1 Current Results

The pipeline is running end to end on Pythia-70m with the suppression intervention.
Quantitative results will be populated here once a full evaluation run completes.

| Metric | Baseline | Targeted | Control |
|---|---|---|---|
| Factual accuracy (TriviaQA EM) | TBD | TBD | TBD |
| Perplexity (WikiText-103) | TBD | — | — |

### 4.2 Observations So Far

Getting the pipeline running exposed several practical challenges that shaped the
current design choices:

**Model scale vs. hardware constraints.** The original plan used Pythia-1.4B, which
requires ~6GB RAM in float32. Development on a CPU-only machine forced a switch to
Pythia-70m (~150MB). The experiment logic is unchanged; the model swap is a single
three-line constant change in `data.py`. Results on 70m will serve as a proof of
concept before scaling up on a GPU.

**SAELens API instability.** The SAELens library has undergone significant API changes
across versions, and the pretrained release names for Pythia models are not consistent
across versions. The correct release name (`pythia-70m-deduped-res-sm`) was identified
by querying the installed library's registry directly rather than relying on
documentation.

**Feature identification is a mean-activation heuristic.** The current approach flags
features that are on average most active across factual QA prompts. This is a
reasonable first approximation but likely captures some domain or syntactic features
that correlate with factual content rather than encoding factual correctness directly.
The control condition provides a partial guard against this, but a contrastive probe
would be more principled.

### 4.3 What Would Change Our Interpretation

The hypothesis predicts: `targeted_accuracy << control_accuracy`, with perplexity
remaining roughly flat. Three alternative outcomes and their implications:

**Targeted ≈ control:** Feature identification is not working. The top-k by mean
activation is the wrong selection criterion. Next step: switch to a contrastive
approach (true vs. false statement activation differences) or target a different layer.

**Both targeted and control drop sharply vs. baseline:** Residual stream interference
from the encode-decode cycle is degrading the model regardless of which features are
zeroed. Next step: use `sae.use_error_term = True` to preserve the original residual
and only subtract the contribution of targeted features.

**Targeted drops but perplexity also spikes:** Factual and fluency features are
entangled at this layer. Next step: try a later layer (layer 5 in Pythia-70m) where
representations are more semantic, or switch from activation patching to weight editing
via `sae.W_dec`.

---

## 5. Remaining Experiments

In rough priority order:

**Quantitative baseline results** — complete a full 100-sample evaluation run and
populate the results table above.

**Contrastive feature identification** — replace the mean-activation heuristic with a
probe trained on the activation difference between true and false answers on the same
questions. This more directly targets features encoding factual *correctness*.

**Layer sweep** — run the suppression experiment at each of the 6 residual stream hook
points in Pythia-70m to see which layer shows the largest targeted/control gap.

**Scale to Pythia-1.4B** — run the full experiment on the originally intended model on
a GPU, using the `pythia-1.4b-deduped` SAE release.

**Noise and swap interventions** — the codebase has hooks for noise injection and
feature swapping already designed; implement and compare against suppression.

---

## 6. Roadblocks

**Hardware.** Running on CPU limits both model size and iteration speed. A single
full evaluation run (100 TriviaQA samples × 3 conditions) takes significant time on
CPU. GPU access would make iteration much faster and enable the originally planned
Pythia-1.4B experiments.

**SAELens API churn.** The library's pretrained registry, import paths, and class APIs
all changed between versions during development. The current code targets SAELens 6.x.

**Feature identification quality.** It is currently unclear whether the top-k
mean-activation approach is identifying features that encode factual knowledge or
features that simply activate on question-answering formatted text. The control
condition provides some signal, but the contrastive probe experiment is needed to
resolve this.

---

## 7. Conclusion (Preliminary)

The full conclusion depends on the remaining experiments, particularly the quantitative
evaluation results and the contrastive feature identification experiment. The key
question the results will answer is whether factual recall and linguistic fluency are
dissociable in Pythia's representation space at the feature level.

If targeted accuracy drops substantially more than control accuracy, that supports the
dissociability hypothesis and validates the SAE-based approach to factual feature
identification. If the gap is small, it suggests either that the feature identification
method needs refinement, or that factual knowledge is more distributed than the SAE
decomposition implies — a finding that would itself be a meaningful contribution to the
interpretability literature.

---

## 8. Contributions

This is a solo project. Steven Dillard is responsible for all components: literature
review, experimental design, implementation, debugging, and writeup.

---

## References

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner,
N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N.,
Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B.,
Burke, J.E., Hume, T., Carter, S., Henighan, T., and Olah, C. "Towards Monosemanticity:
Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread*,
2023. https://transformer-circuits.pub/2023/monosemantic-features

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce,
A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N.L., McDougall, C.,
MacDiarmid, M., Freeman, C.D., Sumers, T.R., Rees, E., Batson, J., Jermyn, A., Carter,
S., Olah, C., and Henighan, T. "Scaling Monosemanticity: Extracting Interpretable
Features from Claude 3 Sonnet." *Transformer Circuits Thread*, 2024.
https://transformer-circuits.pub/2024/scaling-monosemanticity

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika,
M., Dombrowski, A.K., Goel, S., Li, N., Byun, M.J., Wang, Z., Mallen, A., Basart, S.,
Koyejo, S., Song, D., Fredrikson, M., Kolter, J.Z., and Hendrycks, D. "Representation
Engineering: A Top-Down Approach to AI Transparency." *arXiv:2310.01405*, 2023.
https://arxiv.org/abs/2310.01405

Meng, K., Bau, D., Andonian, A., and Belinkov, Y. "Locating and Editing Factual
Associations in GPT." *Advances in Neural Information Processing Systems*, 2022.
https://arxiv.org/abs/2202.05262

Meng, K., Sharma, A.S., Andonian, A., Belinkov, Y., and Bau, D. "Mass-Editing Memory
in a Transformer." *International Conference on Learning Representations*, 2023.
https://arxiv.org/abs/2210.07229

Prasad, A.V., Henighan, T., Batson, J., et al. "Features as Rewards: Scalable
Supervision for Open-Ended Tasks via Interpretability." *arXiv:2602.10067*, 2025.
https://arxiv.org/abs/2602.10067

Bloom, J., Tigges, C., Duong, A., and Chanin, D. "SAELens." GitHub, 2024.
https://github.com/jbloomAus/SAELens
