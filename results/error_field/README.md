# Two fixed-reference response policies

Neither exploratory policy exceeded every control on both backbones.

This is a follow-up method-development experiment with new heldout questions. It is not a confirmed ICML contribution. Its design and models were frozen before inspecting the preceding paired-evidence study's heldout answer-quality results. The development executions are reused; all 800 new test question IDs are excluded from preceding cohorts. Their supporting/filler article titles are absent from the source v3 candidate pools; article-title disjointness from earlier qualification cohorts is not guaranteed. All three input variants stay together in resampling. The benchmark weights conditions equally.

**Supervision:** the error-logit policy uses baseline correctness labels for its probe. Its strategy-response fitting and configuration selection use activation-derived logit changes, without mitigation-success targets. Independent pilot outcomes informed the action bank and the fixed missing-only condition map. Public/confidence transition controls receive the same logit-change targets. The clean-state reference policy uses paired activation targets without correctness labels. Its public/confidence controls receive the same residual/state-change targets. Adding it leaves every error-logit policy parameter and validation prediction unchanged, checked before test. Every test decision is saved before any mitigation preprocessing.

## Test F1 (%) and extra calls

| Backbone | Policy | F1 | Extra calls | Rescued / harmed (EM) |
|---|---|---:|---:|---:|
| qwen7b | Keep | 70.88 | 0 | — |
| qwen7b | Error-logit transitions | 74.35 | 0.590 | 135 / 50 |
| qwen7b | Paired clean-state reference | 71.12 | 0.013 | 9 / 2 |
| qwen7b | Activation condition rule | 73.06 | 0.553 | 138 / 85 |
| qwen7b | Public-feature transitions | 71.28 | 0.013 | 9 / 0 |
| qwen7b | Confidence transitions | 71.09 | 0.007 | 6 / 0 |
| qwen7b | Public condition rule | 72.76 | 0.707 | 149 / 111 |
| qwen7b | Confidence condition rule | 72.49 | 0.095 | 41 / 5 |
| qwen7b | Previous condition transitions | 72.89 | 1.381 | 222 / 172 |
| qwen7b | Public clean-reference transitions | 70.88 | 0.000 | 0 / 0 |
| qwen7b | Confidence clean-reference transitions | 70.88 | 0.000 | 0 / 0 |
| phi35 | Keep | 69.66 | 0 | — |
| phi35 | Error-logit transitions | 71.77 | 0.613 | 111 / 67 |
| phi35 | Paired clean-state reference | 69.93 | 0.028 | 7 / 1 |
| phi35 | Activation condition rule | 70.88 | 0.485 | 102 / 72 |
| phi35 | Public-feature transitions | 71.32 | 0.276 | 70 / 31 |
| phi35 | Confidence transitions | 70.38 | 0.142 | 30 / 14 |
| phi35 | Public condition rule | 70.37 | 0.709 | 121 / 107 |
| phi35 | Confidence condition rule | 70.33 | 0.046 | 17 / 3 |
| phi35 | Previous condition transitions | 68.59 | 1.452 | 161 / 179 |
| phi35 | Public clean-reference transitions | 69.66 | 0.000 | 0 / 0 |
| phi35 | Confidence clean-reference transitions | 69.66 | 0.000 | 0 / 0 |

## Error-logit transitions F1 differences (pp, paired 95% intervals)

| Control | Qwen | Phi |
|---|---:|---:|
| Keep | +3.47 [+2.33, +4.63] | +2.11 [+1.04, +3.17] |
| Best risk-gated fixed within call budget | +1.55 [+0.15, +2.11] | +1.23 [+0.12, +1.83] |
| Shuffle within policy risk bins | +2.01 [+1.20, +2.81] | +2.47 [+1.74, +3.19] |
| Activation condition rule | +1.28 [-0.08, +2.63] | +0.89 [+0.21, +1.60] |
| Public-feature transitions | +3.07 [+1.92, +4.20] | +0.44 [-0.38, +1.29] |
| Confidence transitions | +3.26 [+2.11, +4.39] | +1.38 [+0.39, +2.41] |
| Public condition rule | +1.59 [+0.17, +3.06] | +1.40 [+0.60, +2.21] |
| Confidence condition rule | +1.86 [+0.67, +3.01] | +1.44 [+0.40, +2.47] |
| Previous condition transitions | +1.46 [-0.05, +2.96] | +3.18 [+2.08, +4.26] |
| Public clean-reference transitions | +3.47 [+2.33, +4.63] | +2.11 [+1.04, +3.17] |
| Confidence clean-reference transitions | +3.47 [+2.33, +4.63] | +2.11 [+1.04, +3.17] |

## Paired clean-state reference F1 differences (pp, paired 95% intervals)

| Control | Qwen | Phi |
|---|---:|---:|
| Keep | +0.24 [+0.03, +0.47] | +0.27 [+0.06, +0.50] |
| Best risk-gated fixed within call budget | +0.02 [-0.11, +0.10] | -0.19 [-0.43, +0.02] |
| Shuffle within policy risk bins | +0.18 [-0.01, +0.37] | +0.16 [-0.02, +0.36] |
| Activation condition rule | -1.94 [-3.12, -0.78] | -0.95 [-1.96, +0.13] |
| Public-feature transitions | -0.16 [-0.44, +0.11] | -1.40 [-2.13, -0.64] |
| Confidence transitions | +0.03 [-0.20, +0.26] | -0.45 [-0.99, +0.06] |
| Public condition rule | -1.64 [-2.89, -0.35] | -0.44 [-1.59, +0.72] |
| Confidence condition rule | -1.37 [-1.91, -0.84] | -0.40 [-0.81, -0.03] |
| Previous condition transitions | -1.77 [-3.28, -0.27] | +1.34 [-0.09, +2.77] |
| Public clean-reference transitions | +0.24 [+0.03, +0.47] | +0.27 [+0.06, +0.50] |
| Confidence clean-reference transitions | +0.24 [+0.03, +0.47] | +0.27 [+0.06, +0.50] |

The fixed control optimizes action and risk threshold in hindsight, including unused calls, within the policy's extra-generation-call budget. It also includes one-call sham/empty controls. Matching a call-budget cap does not match tokens, latency or activation-extraction overhead. JSON files include token costs, EM, rescue/harm and prediction timing audits.

## Latent model and transfer assumption

Selected configuration: `{"alpha": 100.0, "features": "bilinear", "kind": "ridge"}`. Shared validation transition NMSE 1.0022 → 0.7255. This predicts an error-probe response, not true strategy success.

Clean-reference configuration: `{"alpha": 100.0, "features": "state", "kind": "ridge"}`; validation state-transition NMSE 0.9324 → 0.6246. The two NMSE objectives measure different quantities and should not be compared directly.

| Backbone | Validation baseline-error AUROC | Test baseline-error AUROC | Oracle action F1 | Empty selections: error / reference |
|---|---:|---:|---:|---:|
| qwen7b | 0.707 | 0.693 | 88.14 | 80 / 0 (of 2400) |
| phi35 | 0.760 | 0.744 | 81.50 | 2 / 0 (of 2400) |

A probe trained on original prompts may be wrong on modified prompts. The per-action probe AUROC/log loss and empty-evidence diagnostic in the JSON files test this assumption; low transition MSE alone cannot validate it. The one-standard-deviation residual penalty is a heuristic and has no calibrated risk guarantee. The clean-reference policy instead requires predicted distance toward its denoised matched-question reference to track repair. Neither assumption follows from transition predictability.

Two exploratory policies were fixed before this fresh cohort was executed. Each must beat every listed control with a positive lower 95% interval bound on both backbones for the same policy. Two-sided 95% bounds correspond to nominal one-sided 2.5% per policy; the Bonferroni family threshold for these two policy claims is 5%, subject to bootstrap approximation. The broader iterative project remains exploratory.

Recoverability is restricted to the available action bank. These data do not identify causal layer pathways, prove universal irrecoverability, establish natural-error transfer, or establish novelty. See [method](../../experiments/error_field/README.md) and the [preceding evidence experiment](../paired_evidence/README.md). Raw states and model objects remain in the ignored run root; selection.json and reference_selection.json contain all configurations and frozen hashes.

## Offline checks and source versions

[Objective diagnostics](OFFLINE_DIAGNOSTICS.md) use actual post-action states and
report additional posthoc confidence/public budget controls. They do not change
the primary acceptance decision. The confidence-budget comparisons do not
establish a replicated activation advantage.

Published packaging includes later search-initialization and output-path fixes.
The original hashes and frozen checkpoints above are intentionally preserved;
see [exact-checkpoint replay](../../REPRODUCE.md#replaying-the-recorded-local-artifacts).
