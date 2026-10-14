# Activation information for choosing a recovery strategy

We ask whether a language model's initial activations can identify a useful
mitigation **without training the strategy selector on which mitigation worked**.
Offline strategy executions provide activation-change targets. The current
error-risk method also uses correctness labels for ordinary baseline answers.

## Current result

The most promising tested method forecasts changes in a frozen baseline-error
readout, then selects keep, retrieve, filter or focus. On 800 fresh questions
with three evidence variants each, per backbone:

| Backbone | Keep F1 | Error-risk policy F1 | Gain (pp, paired 95% interval) |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 70.88 | 74.35 | +3.47 [+2.33, +4.63] |
| Phi-3.5-mini-instruct | 69.66 | 71.77 | +2.11 [+1.04, +3.17] |

It also beat the fixed-action control gated by its own baseline risk score on
both models. **It did not pass the full prespecified control comparison.**
Some differences from simpler learned controls were inconclusive. An additional,
explicitly posthoc confidence-based fixed-action/budget control was competitive
on both backbones. These results do not establish that activations add reliable
value beyond confidence, or constitute an ICML contribution.

Mean extra generation calls were 0.590 and 0.613 per input. This does not include
activation capture/replay overhead and is not a latency comparison.

## Start here

- [Current error-risk method and mathematical assumptions](experiments/error_field/README.md)
- [Fresh response-policy results](results/error_field/README.md)
- [Offline objective diagnostics](results/error_field/OFFLINE_DIAGNOSTICS.md)
- [Original paired-evidence results](results/paired_evidence/README.md)
- [Secondary call-budget check and its failed replication](results/retrieval_budget/README.md)
- [All studies and limitations](RESULTS.md)
- [Reproduction](REPRODUCE.md)

## How the current method works

1. Fit a small error readout from initial prompt activations to **baseline**
   exact-match failure. Freeze it.
2. Learn how each strategy changes that readout, from offline activation pairs.
   Select models by validation transition error, without scoring strategy answers.
3. On a new input, forecast those changes and choose the largest predicted risk
   reduction with a fixed validation residual penalty. Save the decision before
   any candidate mitigation is executed.

The readout must remain meaningful after interventions. Accurate transition
prediction alone cannot establish that. Baseline confidence requires the ordinary
answer generation; all activation states are measured before answer tokens.
The fixed action bank and missing-only condition map were informed by independent
pilot outcomes, so the overall design is not entirely outcome-free.

## What the experiments ruled out

The first paired-evidence study learned condition coordinates and strategy
movements. Neither its condition rule nor its transition policy passed all
controls on both models. Its original same-action clean/corrupted gap score
also has an exact deletion degeneracy: removing evidence makes both inputs
identical, even when both answers are wrong.

A second policy predicted a matched clean state for each new input and kept
that reference fixed while scoring actions. It brought little improvement.
Offline diagnostics show that even actual action states rarely reduce this
penalized distance, so improving its forecaster alone is insufficient.

The simpler missing-evidence detector improved over keep, but a secondary
cost-matched replication did not beat confidence-based routing. Those negative
results are retained alongside the promising error-risk results.

## Repository guide

| Code | Purpose |
|---|---|
| `experiments/paired_evidence/` | Controlled clean/missing/distractor cohorts, collection, activation-only search |
| `experiments/error_field/` | Baseline-error transitions, fixed clean reference, offline diagnostics |
| `experiments/retrieval_budget/` | Recorded secondary budget comparison |
| `experiments/activation_drift/` | Earlier distance-based recovery study |
| `experiments/recovery_scale/` | Earlier success-supervised comparison |
| `experiments/recovery_help/` | Shared backend, prompts and scoring |

Raw states, weights, fitted objects and generations stay outside Git. Compact
results, cohort IDs, configurations, figures, source/model hashes and timing
audits are committed. The new studies use two disjoint 800-base test cohorts;
every condition variant stays with its base question during resampling.

## Scope and next scientific checkpoint

The next requirement is a repair score that remains useful under interventions
and beats uncertainty-based routing on independent questions. The current task
mainly supports retrieval for missing evidence; it does not establish several
distinct internal failure mechanisms with different reliable remedies.

These studies do not identify a defective layer, a causal pathway, natural-error
transfer, universal irrecoverability or novelty. Existing detection/steering
code remains under `scripts/`, `src/` and older results. The
[original overview](https://github.com/samhithaharish/hallucination-mitigation-in-activation-space/blob/04f07bd349e0fd188525bc068d1b3976fcced3d5/README.md)
and [original paired proposal](NEXT_EXPERIMENT.md) are preserved.
