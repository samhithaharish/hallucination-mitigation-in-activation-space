# Pre-registration: NQ-Open replication

Committed before any NQ-Open data was generated. Replicates the three main TriviaQA results on a second dataset,
with the same 10 models, code and decision rules. Nothing is re-tuned on NQ-Open.

## Data

`google-research-datasets/nq_open`, validation split, first 2000 questions. Gold = the list of accepted answers.
The single target string used for edits ("value") is the **shortest** accepted answer. Same chat prompt and the
same answer check (a gold answer appears in the reply). Models with fewer than 150 questions in either class are
excluded, as before.

## Replicated tests (rules copied unchanged from the TriviaQA pre-registrations)

1. `prereg_mixed_vs_separated.md`: M1–M4, the per-model mixed/separated call, S1–S3 (small vs large), and
   A1 (answer-level probe > question-level probe in ≥ 8/10 models).
2. `prereg_type_control.md` (including deviation 1): K1. Gold vs the model's own sampled wrong answer and vs a
   prompted wrong answer, ε = 0.05, 120 questions per model.
3. `prereg_generate_then_select.md`: B1–B3 on the 9 non-development models (Llama-3.2-3B stays the development
   model; nothing is re-developed on NQ-Open).

## What counts as replication

- **Headline (edits reach plausible answers, not true ones):** replicates if, for ≥ 7/10 models, gold is
  **not** significantly easier to reach than the model's own sampled wrong answer (the CI of gold − sampled-wrong
  includes 0 or is below 0).
- **All mixed** replicates if ≥ 8/10 models are called "mixed".
- **A1** and **B1** replicate if they meet their original thresholds.

## Deviations

(none yet)
