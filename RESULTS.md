# What we have found

The goal is to use a model's initial activations to choose a useful recovery
strategy. The stronger version learns strategy effects from activation changes,
without fitting the strategy chooser to mitigation-success labels.

**That stronger claim is not established.** We have completed experiments and
useful measurements, including negative results. The paired evidence study and a
[fresh response-policy follow-up](results/error_field/README.md) are complete.

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

## 3. Paired evidence interventions and activation-only search

For each of 800 fresh base questions per model, the new study constructed clean,
missing-evidence and distractor variants. A separate 400/100 train/validation
split fit paired activation coordinates and strategy-transition predictors.
Supporting titles were separated across these splits. A corrected common
candidate pool removed a detected public membership shortcut; all earlier
qualification questions were excluded. The independent pilot informed the
strategy definitions and the fixed missing-only retrieval map. Learned fitting
and configuration selection used no mitigation-success targets.

| Model | Keep F1 | Learned transitions | Condition rule | Condition minus public-input rule, 95% interval |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 70.71 | 73.30 | 74.78 | +0.76 [+0.17, +1.37] |
| Phi-3.5-mini | 66.44 | 69.13 | 69.49 | +0.03 [−0.62, +0.68] |

The condition rule improved over keep and the strongest risk-gated fixed action
within its call budget on both backbones. Its advantage over public inputs did
not replicate on Phi. Learned transitions did not add reliable value over the
simpler condition rule. Neither policy passed all prespecified comparisons on
both backbones. The bounded autoresearch loop reduced validation activation
NMSE from 0.9674 to 0.4506; that did not establish better repair selection.

The original same-action paired-gap criterion has an exact deletion degeneracy:
empty evidence makes both clean and corrupted prompts identical and maximizes
gap closure. In 280 Qwen and 271 Phi pairs, the clean original answer was correct,
the corrupted answer was wrong, and emptying made both answers wrong while
maximizing the score. These counts concern pairs, not independent questions.
This invalidates that particular repair score, not activation methods generally.

Evidence: [full paired results](results/paired_evidence/README.md),
[comparisons](results/paired_evidence/routing_comparison.png), and
[method](experiments/paired_evidence/README.md).

## 4. Forecasting baseline-error changes and a fixed clean reference

The follow-up reused the 400/100 development executions and reserved another
800 fresh base questions, each with three paired conditions, per backbone.
Its design and model weights were frozen before inspecting the preceding
paired study's heldout answer-quality results. New test supporting/filler titles
were absent from source v3 candidate pools; title disjointness from the older
v2 qualification cohort is not guaranteed.

The error-risk method fits a baseline exact-match error probe, freezes it, and
forecasts its logit changes under each strategy. Those transition targets use
activation states without mitigation-success labels. An alternative denoises
the initial state toward its matched-question clean state and forecasts action
state changes. Its fitting uses no correctness labels. Each method keeps its
own fixed validation residual penalty and was evaluated without test refitting.

| Model | Keep F1 | Error-risk policy | Clean-reference policy | Error-risk minus own-risk fixed control (pp, 95% interval) |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 70.88 | 74.35 | 71.12 | +1.55 [+0.15, +2.11] |
| Phi-3.5-mini | 69.66 | 71.77 | 69.93 | +1.23 [+0.12, +1.83] |

The error-risk policy improves over keep, its own-risk fixed-action control,
and shuffled routing on both models. Its gains over the public condition rule
also have positive intervals on both. However, it does **not** beat every
prespecified control on both models: Qwen's comparisons with the activation
condition rule and earlier condition transitions include zero, as does Phi's
comparison with public-feature transitions. Neither new policy passes the
full criterion. Mean extra calls for the error-risk policy are 0.590/0.613.

A secondary analysis of the simpler missing-evidence detector was recorded
before either new collector completed, after exploratory comparisons on the
first paired test. It matches call budgets and permits hindsight action and
threshold selection from activation/public/confidence rankings. Its fresh
replication failed. This does not revise the first study's primary decision.

An additional **posthoc descriptive** control applies the same strong
public/confidence budget comparison to the error-risk policy. Its differences
from confidence-ranked fixed actions are −0.40 pp [−1.76, +0.77] on Qwen and
+0.22 pp [−0.87, +1.05] on Phi. An activation advantage over that control is
not established. These controls were not fitted deployment policies.

Offline diagnostics use actual post-action activations, unavailable to a
single-input selector. The unpenalized baseline-error readout chooses actions
with F1 78.67 on Qwen but only 69.86 on Phi, versus keep 70.88/69.66. Thus
readout transport across interventions is not uniformly supported. Even with
actual action states, the predicted-clean distance plus its fixed penalty
chooses keep in 2,399/2,400 Qwen and 2,386/2,400 Phi inputs. The clean-reference
objective itself limits intervention, independently of forecasting accuracy.
These are descriptive objective checks, not deployable policy results.

Evidence: [full response-policy results](results/error_field/README.md),
[offline diagnostics](results/error_field/OFFLINE_DIAGNOSTICS.md),
[secondary budget results](results/retrieval_budget/README.md), and
[method/assumptions](experiments/error_field/README.md).

## 5. Earlier causal diagnosis attempt

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

The older summaries describe their original completed runs. The paired study
is a newly executed GPU experiment with frozen model/source hashes and
pre-mitigation decision timestamps in its committed provenance.
