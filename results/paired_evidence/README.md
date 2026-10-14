# Paired evidence repair: heldout results

Neither activation policy met the prespecified routing criterion on both backbones.

Learning strategy changes did not establish added value over the condition rule on both backbones.

Each backbone: 400 training, 100 validation, 800 fresh test base questions; three paired versions per base (2,400 test inputs). The 200-base pilot is separate. The benchmark weights the three conditions equally, not by natural failure prevalence. All test decisions were saved before any mitigation preprocessing. No mitigation-success targets were used for learned parameter fitting or activation-configuration selection. Independent pilot outcomes informed the action definitions and fixed missing-only condition map.

## Test answer F1 (%)

| Backbone | Keep | Best always-on action | Strategy changes | Condition rule | Public input | Confidence |
|---|---:|---:|---:|---:|---:|---:|
| qwen7b | 70.71 | 71.90 | 73.30 | 74.78 | 74.02 | 72.81 |
| phi35 | 66.44 | 67.92 | 69.13 | 69.49 | 69.47 | 67.17 |

Best always-on action is chosen in hindsight. Matched-risk fixed controls below optimize action and risk threshold in hindsight within the same extra-generation-call budget, including cheaper sham and empty controls and unused calls; this does not match tokens or latency.

## Paired F1 differences (percentage points, 95% intervals)

Intervals resample 800 base questions, retaining all three variants. Positive lower bounds are required against **every** listed control and on both backbones for the same policy.

| Backbone / policy | Keep | Matched-risk fixed | Risk-bin shuffle | Public input | Confidence |
|---|---:|---:|---:|---:|---:|
| qwen7b / Learned strategy changes | +2.59 [+0.95, +4.17] | -0.80 [-1.80, -0.04] | +0.71 [-0.04, +1.38] | -0.72 [-1.91, +0.43] | +0.49 [-1.10, +2.02] |
| qwen7b / Condition rule | +4.07 [+2.86, +5.26] | +2.03 [+0.95, +2.97] | +2.62 [+1.88, +3.29] | +0.76 [+0.17, +1.37] | +1.97 [+0.83, +3.11] |
| phi35 / Learned strategy changes | +2.69 [+1.28, +4.07] | +0.23 [-0.86, +1.16] | +1.57 [+0.79, +2.29] | -0.33 [-1.41, +0.77] | +1.96 [+0.61, +3.32] |
| phi35 / Condition rule | +3.05 [+2.09, +4.03] | +2.34 [+1.19, +3.12] | +2.07 [+1.39, +2.70] | +0.03 [-0.62, +0.68] | +2.32 [+1.33, +3.27] |

Strategy changes versus the condition rule: qwen7b -1.48 [-2.57, -0.42]; phi35 -0.36 [-1.39, +0.63].

## Activation-only search

Twelve ridge configurations were attempted plus the unconditional mean baseline; 0 trials crashed. Mean validation activation NMSE: 0.9674 → 0.4506. Selected trial_07; configuration: `{"alpha": 100.0, "features": "state", "kind": "ridge"}`. This score measures predicted activation changes, not answer improvement.

| Backbone | Validation condition accuracy | Test condition accuracy | Mean extra calls: strategy / condition |
|---|---:|---:|---:|
| qwen7b | 94.67 | 92.79 | 1.374 / 0.568 |
| phi35 | 90.67 | 87.04 | 1.402 / 0.456 |

## A failed geometry criterion

Transforming both a corrupted and a matched clean prompt with the same action, then minimizing their activation distance, has a deletion-collapse degeneracy. Emptying the evidence makes both prompts identical and therefore maximally closes that distance, regardless of answer truth. Counterexamples start with a correct clean answer and incorrect corrupted answer, yet emptying makes both answers wrong while maximizing the gap score. This is a failure of this specific score, not a proof that all activation geometry is useless.

| Backbone | Empty is a gap-score maximizer (%) | Empty F1 | Original corrupted-input F1 | Both wrong counterexamples |
|---|---:|---:|---:|---:|
| qwen7b | 100.00 | 28.03 | 65.35 | 280 |
| phi35 | 100.00 | 27.46 | 60.85 | 271 |

The learned score with sham/empty controls allowed (diagnostic only): qwen7b empty selections 0 / 2400, F1 73.52; phi35 empty selections 0 / 2400, F1 69.11.

## What these results establish

The benchmark controls evidence interventions; its labels describe input conditions. They do not identify a defective layer or prove an internal failure mechanism. Natural-error transfer, other tasks, other backbones, semantic controls and stronger representation-based observable baselines remain necessary. An unrepaired output is not proven irrecoverable.

Activation-based retrieval and evidence-condition classification have prior work; see [method and related work](../../experiments/paired_evidence/README.md). The two-model acceptance result above is the decision for this experiment; no test outcomes were used to refit a winning policy.

Files: *_test.json contains all answer metrics, paired controls, call/token costs and timing audits; *_pilot.json records separate feasibility checks; cohorts.json and qualification_cohorts.json identify fresh and excluded exploratory cohorts; search.json and provenance.json record activation-only selection and frozen source/model hashes. Raw states, generated answers and fitted objects stay in the local ignored run root.

An earlier queued search loaded a stale trial grid and was stopped before heldout collection. Its attempted trials were preserved; the committed grid was then run afresh. Both records are included in search.json.

The search initialization was hardened after this run so reproduction can start
from the published winning predictor configuration. The recorded source hashes
and Git checkpoint remain those of the actual frozen run, not the patched
packaging version. See [reproduction](../../REPRODUCE.md).
