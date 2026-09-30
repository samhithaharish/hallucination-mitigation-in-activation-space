# Pre-registration: type-matched control for "the correct answer is nearby"

Committed before any data for this test was generated.

## Why

In the pilot and the 10-model sweep (M3c), a small edit reached the gold answer far more often than a
**random other question's** answer. Two confounds could explain that without any knowledge:

1. **Type:** a random answer is often the wrong type (a year for a "who" question), which the model may
   resist regardless of knowledge.
2. **Aliases:** gold success accepted any alias in the gold list; control success accepted one string.

## Design

- Models: the same 10 as `prereg_mixed_vs_separated.md`; records reused from that sweep.
- Questions: 120 greedy-wrong train questions per model (seed 2), edit layer round(0.45 × n_layers),
  30 Adam steps, ε ∈ {0.05, 0.1}, as before.
- Targets, each scored the same way: **success = the greedy output contains that exact target string**
  (after normalization). No alias lists for any condition.
  - **gold:** the TriviaQA answer value.
  - **sampled-wrong (type-matched, model-plausible):** the most frequent of the model's own sampled answers that
    is incorrect and differs from its greedy answer.
  - **prompted-wrong (type-matched):** the model's greedy reply to "Give one plausible but incorrect answer",
    kept only if incorrect and different from the greedy answer.
  - **random:** another question's answer value (the old control, now scored the same way).
- Target length in tokens is recorded for every condition.

## Hypothesis

**K1:** at ε = 0.05, gold success > sampled-wrong success **and** gold > prompted-wrong success (paired
bootstrap over questions with both targets, 95% CI > 0) in ≥ 8 of 10 models. If K1 fails, the claim "the
correct answer is nearby" is withdrawn and replaced with "any type-appropriate answer is nearby".

Secondary: the same at ε = 0.1, and a length-matched check (restricted to pairs whose token lengths differ
by ≤ 1).

## Deviations

(none yet)
