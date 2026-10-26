# Pre-registration: comparison with Jin et al. (arXiv 2509.06472) and SeaKR (arXiv 2406.19215)

Written 2026-10-23, before any of the data below exists. Same 500 test questions (qi 500-999), candidates and
judged answers as `docs/prereg_context_select.md`; models Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-3B.

## What the prior methods do (from the papers)

- **Jin et al.:** a classifier on the hidden state at the middle layer (L/2) before the first answer token,
  trained on answer correctness; a passage's value is Conf(Q + C_i) - Conf(Q), each passage scored on its own.
  At test time the probe is distilled into a fine-tuned bge-reranker-v2-m3 (not released). Reported gains:
  +4.7 pts on NQ, +0.8 to +1.4 on HotpotQA (Llama-3-8B).
- **SeaKR:** for each candidate passage, sample k = 20 generations, take the EOS-token hidden state at layer
  L/2, and score uncertainty as the log-determinant of the regularized Gram matrix (following INSIDE); pick the
  least uncertain. N = 3 candidates. Reported re-ranking gain: +1.1 to +2.8 F1.

## Hypothesis

Our gain is larger mainly because we score the **pair** jointly (multi-hop questions need two passages that
are only useful together), not because of the layer or the classifier.

## Rankers (each picks a pair of paragraphs; the answer is the already-judged greedy answer for that pair)

1. **Ours (as published in the context-selection pilot):** validation-selected layer, pair scoring.
2. **Jin-style, single passages:** MLP probe (one hidden layer of 256, early stopping) trained on the 6,000
   baseline `train` states at layer L/2, label = keep F1 >= 0.5; score each paragraph alone (question + that
   paragraph, Som's keep prompt); pick the top-2 singles. Conf(Q) is constant within a question, so ranking by
   Conf(Q + C_i) equals ranking by the shift.
3. **Ours, single passages:** ranker 1's probe applied to single paragraphs, top-2 singles.
4. **Middle layer, pairs:** ranker 1's logistic probe re-fitted at layer L/2, pair scoring.
5. **bge-reranker-v2-m3** (Jin's base reranker, not fine-tuned): top-2 paragraphs by score.
6. **SeaKR-style (Llama-3.2-3B only):** for each pair, k = 8 samples (temperature 0.7, top-p 0.9; k = 8
   instead of 20 for cost, logged here in advance); hidden state of the last generated token at layer L/2 via a
   teacher-forced pass; score = (1/k) * sum log(eig(Sigma + 1e-3 I)) with Sigma = Z^T J Z over the k
   embeddings (INSIDE's covariance with feature centering); pick the minimum.

## Comparisons (judged accuracy, Qwen2.5-7B judge primary, Laya secondary; paired bootstrap 95% CI)

- **J1 (pair vs single):** ranker 1 minus ranker 3 > 0 in at least 2 of 3 models.
- **J2:** ranker 1 minus ranker 2 (Jin-style) > 0 in at least 2 of 3 models.
- **J3:** ranker 1 minus ranker 5 (bge) > 0 in at least 2 of 3 models.
- **J4:** ranker 1 minus ranker 6 (SeaKR-style) > 0 on Llama-3.2-3B.
- Descriptive: ranker 1 minus ranker 4 (layer choice); compute per question for each ranker (forward passes,
  generations).

Failed comparisons are reported as results. Deviations are logged below with dates.

## Deviations

(none yet)
- 2026-10-26 (before any Llama-3.1-8B data exists): extended to Llama-3.1-8B-Instruct with the same protocol. Its
  probe is trained on 6,000 `train` / 1,000 `validation` baseline answers collected without the recovery actions
  (`collect.py --keep-only`; same keep prompt and states). Probe layer chosen on validation as before; captured
  pair layers = {L/2, axis-3, axis, axis+3}. Qwen2.5-7B stays the primary judge (no family overlap); Laya secondary.
