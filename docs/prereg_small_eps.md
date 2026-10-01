# Pre-registration: smaller edit budget for the large models

Committed before any data for this run was generated.

## Why

At ε = 0.05 the 7–8B models reach almost any target (gold 79–97%, sampled-wrong 85–100%, and even unrelated
answers up to 47% on Granite-8B), so the gold-vs-plausible-wrong comparison is near ceiling. A smaller budget
should separate them.

## Design

- Models: the 5 large ones (Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B, Granite-3.1-8B, Mistral-7B), both datasets.
- Same 120 questions, layer, targets and optimizer as `prereg_type_control.md` (same seed). Conditions: gold,
  sampled-wrong, random (prompted-wrong is skipped).
- ε ∈ {0.01, 0.02, 0.03}.
- Analysis drops every sampled-wrong target that either judge in `prereg_judge_audit.md` calls correct.

## Decision rule

For each dataset, pick the **pre-specified** reference budget ε\* = the largest ε in {0.01, 0.02, 0.03} at which
the random-answer success rate is ≤ 15% in every large model (so the test is off the ceiling). At ε\*, the headline
holds for the large models if gold is **not** significantly easier than sampled-wrong in ≥ 4/5 models (paired
bootstrap, 95% CI). If no ε meets the random ≤ 15% condition, report all three and draw no conclusion.

## Deviations

(none yet)
