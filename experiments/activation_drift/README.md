# Activation-drift experiment

Can activation changes predict a useful recovery action without fitting on
that action's success labels? The completed pilot did not establish superiority
over strong fixed-strategy controls. See [results](../../RESULTS.md).

## Recorded design

- Frozen Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct.
- HotpotQA distractor: 800 calibration, 200 validation, 1,000 test questions per
  model. Eight additional smoke-test questions are excluded; two were used per
  model for capture checks. Seed 20261010.
- Exclude 12,624 questions from earlier recovery studies. Test supporting article
  titles are excluded from fitting/validation; validation titles from fitting.
- Actions: keep, retry, retrieve, reason, both, fresh, prune. Active actions each
  cost one generation, with different token usage. Evidence: top 2 initially,
  top 5 when expanded. Direct cap 64; reasoning cap 256.
- Greedy vLLM generation. Transformers replay measures exact recorded token
  prefixes at six fractional depths, at prompt end and immediately before the
  last nonterminal output token. That output token can be closing markup.
- Fit 16 PCA components per layer/time using training baseline states. Scale by
  component standard deviation with a floor of 0.1 times the largest standard
  deviation. No test fitting.

## Rule

Let `p` be the compressed prompt state and `e` the compressed endpoint. Fit
`F(p)` to endpoints of initially correct calibration answers, using ordinary
baseline exact-match labels. Define `r = e - F(p)`. This is an estimated typical
correct endpoint, not the known correct state for the same question.

For each strategy, fit its observed displacement `e_a - e` from `[p, e]` using
ridge regression. Choose regularization by activation-displacement MSE on
validation data. The mean-displacement variant uses a constant vector instead.

```text
score(a) = (||r||² - ||r + predicted_displacement(a)||²) / dimension
score(keep) = 0
choice = argmax(score)
```

No strategy-success labels enter fitting or tuning. Baseline correctness is
used, so the method is not fully unsupervised. Conditional displacement models
use the initial states as well as the residual; they are activation-only, not
strictly drift-only predictors. A fresh test question requires an initial
answer and its captured/replayed states, without trying remedies first.

Controls include mean, reversed, shuffled and randomly oriented moves;
incorrect/permuted-label references; fixed strategies; and routing shuffled
within baseline-risk bins. Evaluation reports rescue, harm, F1, calls, tokens,
question-bootstrap intervals and supporting-article component intervals.

## Supplement

The primary score omits displacement uncertainty. A separate supplement
subtracts the validation displacement MSE with a fixed multiplier of one.
For the original experiment this was frozen before test scoring, after some
test generation had begun; it was not part of the original primary protocol.
The reproduction commands freeze it after calibration and before test collection.
It is a pooled error penalty, not an exact conditional variance estimate.

## Files and execution

| File | Purpose |
|---|---|
| pilot.py | Data checks, exact-token capture, geometry fitting, test choices |
| analyze.py | Scoring, comparisons, statistical controls and plots |
| supplement.py | Fixed displacement-error penalty |
| audit.py | Reconstruct prompts/answers and verify stored arrays and identities |
| tests/ | Geometry, token alignment and statistical regression tests |

Use [REPRODUCE.md](../../REPRODUCE.md). Run these as Python modules from the
repository root. Output goes under `runs/activation_drift`, outside tracked
source. Setup freezes hashes of this packaged implementation; original run
summaries remain separate under `results/activation_drift`.
