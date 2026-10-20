# Pre-registration: does added input text move activations along the correct/wrong axis, and does the answer follow?

Written 2026-10-20, before any data below exists. Pilot model: Llama-3.2-3B-Instruct. Builds on the
layer-separability collection (`docs/prereg_layer_separability.md`) and Som Sagar's recovery prompts.

## Question

The correct/wrong "gap" is the direction a linear probe uses to separate correct from wrong initial answers.
When we add text to the input (evidence, metadata, a warning), does the internal state move along that
direction, and when it moves towards "correct", does the answer actually become correct (graded by an LLM
judge)? This is the input-side version of FACR's activation-edit question: detectable movement vs actual fix.

## Data

- Questions: the first 1,000 of Som's frozen `test` cohort (never used to fit the probe).
- Probe axis: fit on the 6,000 `train` baseline states already collected (`runs/layer_scaling/llama32_3b`),
  label = keep-answer F1 >= 0.5, logistic regression as in the separability study, **prompt-end position**
  (added text changes the prompt, and this state precedes the answer). Layer and L2 strength chosen on the
  1,000 `validation` questions. Answer-end drift is reported as secondary.

## Conditions (greedy, answer cap 64 tokens, Som's keep-prompt format)

| Key | Change to the base prompt | Role |
|---|---|---|
| `base` | Som's keep prompt, top-2 BM25 passages | reference |
| `more_evidence` | top-5 BM25 passages | informative |
| `oracle_evidence` | the gold supporting paragraphs plus the top-2 non-supporting BM25 passages | positive control |
| `metadata` | a line "Question type: {bridge/comparison}. Difficulty: {level}." before the question | informative |
| `warning` | a line "Be careful: questions like this are often answered incorrectly. Check each step against the evidence." | informative |
| `filler` | top-2 BM25 passages plus 3 paragraphs from a random other question | length control for `more_evidence` |
| `misleading` | a line "Hint: the answer might be {X}." with X = the opposite of yes/no for yes/no questions, else a random other question's gold answer | negative control |

States at every layer (prompt end and last answer token) for every condition.

## Outcome grading

Primary: LLM judge, Qwen2.5-7B-Instruct, same yes/no prompt and logit rule as FACR's `judge_audit.py`.
Secondary: Som's answer F1 >= 0.5.

## Measures

- Drift d(i, c) = probe logit at the condition's state minus probe logit at the `base` state (positive = towards
  "correct").
- Outcome change: judged-correct under c minus judged-correct under `base`.

## Hypotheses and pass criteria

- **D1 (conditions move as expected):** mean drift is > 0 for `oracle_evidence` and < 0 for `misleading`,
  each with a bootstrap 95% CI excluding 0.
- **D2 (drift predicts fixing, primary):** among questions judged wrong at `base`, pooled over the four
  informative/positive conditions, AUROC of d(i, c) for "judged correct under c" has a 95% CI above 0.5.
  The same AUROC under `filler` is reported as the length control.
- **D3 (actionability gap, descriptive):** among base-wrong questions whose state crosses to the correct side
  of the probe (logit from < 0 to > 0), the share whose answer is still judged wrong; reported per condition.

A pass on D1 with a failure on D2, or a large D3, supports "detectable, not actionable" on the input side.
Failed hypotheses are reported as results. Deviations are logged below with dates.

## Deviations

(none yet)
