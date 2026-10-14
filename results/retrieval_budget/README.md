# Missing-evidence routing at a fixed call budget

The unchanged activation condition rule did not pass every secondary control on both fresh backbones.

**Secondary exploratory analysis.** The preceding paired study failed its original primary acceptance criterion. This narrower call-budget hypothesis arose after inspecting its results. Its code, controls, policy and fresh cohort/model hashes were registered during the new collection, before either collector completed and before fresh answer outcomes were inspected. It does not replace either preceding study's primary decision. Nominal intervals do not control multiplicity over the broader project.

## Detector and costs

The detector is the already frozen condition rule: retrieve for predicted missing evidence, otherwise keep. It is not trained on mitigation-success targets. The independent pilot informed the action bank and fixed map. The action bank contains keep, retrieve, filter, focus, sham and empty. All three input variants have equal benchmark weight.

| Cohort | Backbone | Keep F1 | Activation F1 | Public default F1 | Extra calls: activation / public default |
|---|---|---:|---:|---:|---:|
| exploration | qwen7b | 70.71 | 74.78 | 74.02 | 0.568 / 0.696 |
| exploration | phi35 | 66.44 | 69.49 | 69.47 | 0.456 / 0.689 |
| replication | qwen7b | 70.88 | 73.06 | 72.76 | 0.553 / 0.707 |
| replication | phi35 | 69.66 | 70.88 | 70.37 | 0.485 / 0.709 |

## Exploration differences (F1 pp, paired 95% intervals)

| Control | Qwen | Phi |
|---|---:|---:|
| Keep | +4.07 [+2.86, +5.26] | +3.05 [+2.09, +4.03] |
| Fixed action + activation risk | +2.03 [+0.95, +2.97] | +2.34 [+1.19, +3.12] |
| Fixed action + public margin | +1.27 [+0.58, +1.93] | +0.84 [+0.08, +1.49] |
| Fixed action + confidence margin | +1.27 [-0.05, +2.21] | +0.15 [-0.82, +0.98] |
| Shuffle within activation risk | +2.62 [+1.88, +3.29] | +2.07 [+1.39, +2.70] |

## Replication differences (F1 pp, paired 95% intervals)

| Control | Qwen | Phi |
|---|---:|---:|
| Keep | +2.18 [+1.04, +3.38] | +1.22 [+0.14, +2.25] |
| Fixed action + activation risk | +0.46 [-0.59, +1.44] | +1.13 [-0.12, +1.86] |
| Fixed action + public margin | +0.24 [-0.49, +0.92] | +0.74 [-0.30, +1.22] |
| Fixed action + confidence margin | -1.56 [-3.22, -0.16] | -0.67 [-1.75, +0.18] |
| Shuffle within activation risk | +1.60 [+0.86, +2.28] | +1.30 [+0.63, +2.03] |

Each fixed control selects its action and best affordable prefix **in hindsight**. It can leave calls unused and can use one-call sham/empty actions on more inputs. Prefixes rank activation condition drift or frozen public/confidence missing-condition margins. The budget cap equals the activation detector's extra call count. Action, threshold and budget are recomputed in every base-question bootstrap. This is an evaluation control, not success-supervised fitting of the deployed rule.

Matching extra generation calls does not match tokens, latency or activation capture/replay overhead. The observable comparator is the recorded TF-IDF/SVD, confidence, length and public-pool model, not every possible text model. These controlled evidence interventions do not establish natural-error transfer or causal layer diagnosis. Activation-based retrieval and evidence-condition probing have prior work; this result cannot establish novelty.

The fresh analysis requires positive lower 95% bounds against every listed control on both models for the same frozen policy. See [registration](registration.json), [method and reproduction](../../experiments/retrieval_budget/README.md), [original paired results](../paired_evidence/README.md), and [separate response-policy results](../error_field/README.md).
