# Pre-registration: detector-gated generate-then-select with calibrated abstention (Path B)

Committed before any `scores.npz` existed. Data come from the mixed-vs-separated sweep
(`prereg_mixed_vs_separated.md`): per question, greedy answer + 10 samples at T = 1; the question-level
detector p_h (layer = M1 layer); per unique candidate answer: the answer-level probe p (depth chosen by
grouped CV), its frequency among the 11 answers, its mean token log-prob, and whether it is the greedy answer.
All train-split scores are out-of-fold, so every choice below is made on train and applied once to test.

**Development vs evaluation.** Llama-3.2-3B is the development model (its naive probe-selection result was
seen: −4.3 pts vs greedy). The other 9 models are held out for evaluation.

## Method

1. **Candidate confidence s(c):** logistic regression on [logit(probe p), frequency, log-prob, is-greedy], fit
   on train candidates (out-of-fold features, correctness labels), giving calibrated p(correct).
2. **Switch rule:** answer the top candidate b instead of the greedy answer g only if p_h(q) > τ and
   s(b) − s(g) > m. τ ∈ quantiles {0.5, 0.6, 0.7, 0.8, 0.9} of train p_h; m ∈ {0, 0.05, 0.1, 0.2, 0.3};
   both chosen on train to maximize accuracy.
3. **Abstain rule:** answer only if s(answer) > λ/(1+λ).

## Metrics (test split)

- Accuracy (always answering): switch rule vs greedy and vs majority vote.
- Reliability R_λ = (#correct − λ·#wrong)/N for λ ∈ {1, 2}, for ours (switch + abstain) vs these baselines,
  each with its threshold tuned on train:
  - (a) greedy, always answering
  - (b) greedy + detector abstention (abstain if p_h > t)
  - (c) greedy + self-consistency abstention (abstain if the greedy answer's frequency < t)
  - (d) greedy + calibrated abstention (abstain if s(g) < λ/(1+λ))
  - (e) majority vote + self-consistency abstention
- Paired bootstrap over test questions (2000 resamples, 95% CI).

## Hypotheses (9 held-out models)

- **B1, selection helps accuracy:** the CI of (switch − greedy) > 0 in ≥ 6/9 models, and the switch rule
  ≥ majority vote in ≥ 6/9.
- **B2, beats the abstention baselines:** for λ = 1 and λ = 2 separately, the CI of (ours − best baseline)
  > 0 in ≥ 6/9 models, where the best baseline is the one with the highest train reliability.
- **B3, the knowledge comes from selection, not only abstention:** at matched coverage (ours vs (c) at the same
  answer rate), ours has higher selective accuracy in ≥ 6/9 models.

## Deviations

(none yet)
