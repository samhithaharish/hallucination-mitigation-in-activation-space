# Offline objective diagnostics

**These are not deployable policies.** Every row below uses actual post-action activations. The named actual-clean oracle additionally receives the matched clean baseline state. These descriptive comparisons do not change any fitted policy or acceptance decision.

| Objective using actual action states | Qwen F1 | Phi F1 |
|---|---:|---:|
| Keep | 70.88 | 69.66 |
| Actual error readout | 78.67 | 69.86 |
| Actual error readout + penalty | 76.90 | 71.44 |
| Actual distance to predicted clean | 71.19 | 70.28 |
| Actual distance to predicted clean + penalty | 70.89 | 69.79 |
| Actual distance to actual clean (oracle) | 73.33 | 72.12 |

Different objective policies may use different actions and call counts; their F1 differences alone do not isolate forecasting error at equal cost. These comparisons test whether a frozen objective is useful even when its action-state forecast is replaced by the actual post-action state. They do not establish a causal error mechanism.

With actual action states and the fixed reference-distance penalty, keep is selected in 2399/2400 qwen7b, 2386/2400 phi35 inputs. This shows the estimated-reference objective itself can discourage nearly all interventions, even when action forecasting is removed.

## Additional fixed-budget controls

These **posthoc descriptive** controls evaluate the unchanged error-risk policy against the best single action and affordable threshold in hindsight, ranked by the frozen public/confidence missing-condition margin. Cheaper sham/empty actions and unused calls are allowed. Every bootstrap retains the three versions of a base question.

| Gate for fixed-action control | Qwen difference (pp, 95% interval) | Phi difference |
|---|---:|---:|
| public | +1.52 [+0.20, +2.73] | +1.37 [+0.48, +2.00] |
| confidence | -0.40 [-1.76, +0.77] | +0.22 [-0.87, +1.05] |

An advantage over the confidence-budget control is not established on both backbones. These posthoc controls do not revise the main policy acceptance result. Extra-call matching does not match tokens, latency or activation-extraction overhead.

Raw details, selected action counts, source/model hashes and timing audits: [Qwen](qwen7b_offline_diagnostics.json), [Phi](phi35_offline_diagnostics.json). See the [main results](README.md) and [method](../../experiments/error_field/README.md).
