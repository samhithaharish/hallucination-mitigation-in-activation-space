# What we have found

The goal is to use a model's initial activations to choose a useful recovery
strategy. The stronger version learns strategy effects from activation changes,
without fitting the strategy chooser to mitigation-success labels.

**That stronger claim is not established.** We have completed experiments and
some useful measurements, including negative results. The proposed paired
experiment in [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) has not been run.

## 1. Predicting recovery from labelled examples

Two frozen models, Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct, were evaluated
on HotpotQA distractor questions. Each model used 6,000 training, 1,000 validation,
2,000 matched test and 1,000 shifted test questions. The ten choices were keep,
retry, retrieve, reason, retrieve-and-reason, fresh answer, prune evidence,
generate a retrieval query, verify, and vote. Offline collection used 280,000
main generations plus 7,168 repeated generations.

This experiment **did train on each strategy's answer-quality gain**. It tests
whether activations add predictive information under that supervision; it does
not test the current label-free strategy-direction idea. Here, "label-free"
refers to strategy outcomes, not to baseline correctness.

F1 below is answer token F1 on a 0–100 scale. These are separate test cohorts
from the activation-drift study, so their absolute scores are not directly
comparable.

| Model | Initial answer | Behavioral selector | Activation selector | Activation minus behavioral, 95% interval |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 59.50 | 69.32 | 71.57 | +2.26 [+1.01, +3.53] |
| Phi-3.5-mini | 55.86 | 63.90 | 63.03 | −0.87 [−2.03, +0.27] |

The primary comparison allocated one additional generation per question on
average. Behavioral methods used confidence and/or text features; candidates
and hyperparameters were chosen on validation data. Activations helped Qwen,
but the positive result did not replicate on Phi. Expanding the remedy menu
helped both models, including their behavioral selectors.

Evidence: [recorded summary](results/recovery_scale/summary.json) and
[comparison plot](results/recovery_scale/policy_frontier.png).
Code: [experiments/recovery_scale](experiments/recovery_scale).

## 2. Choosing a strategy using activation drift

This study used 800 calibration, 200 validation and 1,000 fresh test questions
per model, excluding earlier recovery-study questions. Test supporting article
titles were excluded from training and validation. Six active strategies plus
keep produced 28,000 main generations across the two models.

The method compressed activations at six depths, estimated a typical endpoint
from initially correct calibration answers, and predicted how each strategy
would change the endpoint. It chose the strategy predicted to reduce the
distance to that reference. Strategy-success labels were not used to fit or tune
this method. Baseline correctness labels were used to fit the reference.

| Model | Initial answer F1 | Conditional-direction F1 | Strongest fixed strategy at matched call count | Difference, 95% interval |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 61.54 | 66.05 | 66.73 | −0.68 [−2.41, +1.10] |
| Phi-3.5-mini | 56.78 | 63.67 | 64.14 | −0.47 [−2.02, +1.06] |

The matched control applies one strategy to the same number of questions,
ordered by baseline drift magnitude. Its strategy is selected in hindsight as
a conservative evaluation benchmark: retrieve-and-reason for Qwen, fresh answer
for Phi. Mean additional calls were 0.537 and 0.996, respectively. Equal call
counts do not imply equal token costs or latency.

The directional policy improved over keeping every initial answer. It did not
establish an advantage over this stronger fixed-strategy control. The two-model
primary success criterion was not met. A supplementary penalty for displacement
prediction error reduced calls but did not establish a routing advantage.

Evidence: [recorded summary](results/activation_drift/summary.json) and
[comparison plot](results/activation_drift/geometry_results.png).
Code and method: [experiments/activation_drift](experiments/activation_drift).

### What the repeatable directions mean

For each strategy, we averaged the activation change over two halves of the
800-question calibration set. The two means had cosine similarities of about
0.88–0.997. This is one split-half check in a shared compressed coordinate
system, not replication on a new task.

However, the squared mean shift accounted for only about 1–30% of the average
squared individual displacement. This ratio is not the variance explained by
projection onto that direction. Different strategies also had similar means:
for example, reason and retrieve-and-reason were nearly aligned.

These observations show repeatable average changes under fixed procedures.
They do not establish unique strategy fingerprints or failure-specific repair
directions. Prompt wording, evidence changes and generated answer format can
all contribute. The measurement near the end of an answer often precedes
closing markup rather than the semantic answer itself.

## 3. Earlier causal diagnosis attempt

An earlier controlled color/shape task tried targeted activation patches on
Qwen-7B, Qwen-3B and Granite-2B. The learned patches rarely produced the intended
semantic counterfactual, and the crossed repair-specificity criterion failed.
No reproducible causal pathway across layers was established. That unsuccessful
patching pipeline is not needed to run the two recovery studies packaged here.

The original detection and steering study already exists in this repository's
`scripts/`, `src/` and earlier `results/` folders. Its overview is preserved in
the [previous README](https://github.com/samhithaharish/hallucination-mitigation-in-activation-space/blob/04f07bd349e0fd188525bc068d1b3976fcced3d5/README.md).

## What remains unproven

- Activation drift distinguishes internal causes of failure.
- Moving toward a correct-associated region causes an answer to improve.
- Drift reliably identifies the best remedy from a single initial run.
- The relationship transfers to natural errors, other tasks or unseen strategies.
- Failure under the tested action menu implies that an answer cannot be repaired.

The saved summaries describe the original completed runs. This branch packages
their code with portable paths and setup; it does not claim a new model run or a
positive result from the next experiment.
