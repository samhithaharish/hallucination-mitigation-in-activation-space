# Pre-registration: are hallucinations mixed with or separated from knowledge? (5 small vs 5 large models)

Committed before any data for this sweep was generated. Motivated by a question from the advisor: internally,
are hallucinated answers mixed with real knowledge or separated from it, does this differ between small and
large models, and if they are mixed, how do we fix it?

## Models (instruction-tuned; Gemma excluded because gated)

- **Small:** Llama-3.2-3B, Qwen2.5-3B, OLMo-2-1B (0425), Granite-3.1-2B, SmolLM2-1.7B
- **Large:** Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B (1124), Granite-3.1-8B, Mistral-7B-v0.3

## Per-model protocol (`scripts/mixed_vs_separated.py`)

- **Data.** The first 2000 TriviaQA rc.nocontext validation questions, the same chat prompt and the same
  containment label rule as `prereg_go_nogo.md`. A model is kept only if both classes have ≥ 150 questions.
  Split 70/30 stratified (seed 0).
- **Samples.** Greedy answer + 10 samples at T = 1.

**M1, question-level detectability.** Probe (StandardScaler + LR, C = 0.05) on the mean-pooled prompt state
at every layer, 5-fold CV on train. M1 = max AUROC over layers; also report the argmax layer as a fraction
of depth.

**M2, knows fraction.** Among greedy-wrong questions, the share with ≥ 2/10 correct samples.

**M3, shared correction direction.** 60 greedy-wrong train questions (seed 1), edit layer = round(0.45 ×
n_layers). Per-question gold edits as in the correctability pilot (30 Adam steps, ε ∈ {0.05, 0.1}), plus
random-answer control edits.
- M3a = mean pairwise cosine of the ε = 0.1 gold edits.
- M3b = transfer ratio: fix rate of the half-split mean edit ÷ per-question fix rate, at ε = 0.1.
- M3c = gold − control fix rate at ε = 0.05.

**M4, answer-level separability.** Candidates = the unique (normalized) greedy answer and samples. Each
candidate is teacher-forced after the prompt; feature = mean over answer tokens at 7 depths {0.25, 0.4,
0.5, 0.6, 0.75, 0.9, 1.0}. Probe with question-grouped 5-fold CV on train candidates picks the depth.
M4 = test AUROC (correct vs incorrect candidates) at that depth.

**F, the proposed fix: generate then select.** For each test question, pick the candidate with the highest
probe p(correct). Compare with greedy accuracy, majority vote over the 11 answers, and the oracle (any
correct candidate). Paired bootstrap over test questions (2000 resamples).

## Hypotheses

- **S1, scale → separation (detection):** M1 is higher for large than small models (one-sided Mann-Whitney
  over the 5 vs 5 models, p < 0.05).
- **S2, scale → knowledge:** M2 is higher for large models (same test).
- **S3, scale → shared direction:** M3b is higher for large models (same test).
- **Mixed/separated call per model:** "separated" if M1 ≥ 0.80 **and** M3b ≥ 1/3, otherwise "mixed".
- **A1, knowledge separates at the answer level:** M4 > M1 in ≥ 8 of 10 models.
- **A2, the fix works:** probe-selected accuracy exceeds both greedy and majority vote (95% CI of each
  difference > 0) in ≥ 7 of 10 models.

## Deviations

(none yet)
