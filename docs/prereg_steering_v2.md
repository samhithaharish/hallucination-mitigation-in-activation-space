# Pre-registration v2: stronger steering (Llama-3.2-3B, TriviaQA)

Committed before any v2 steering data was generated. Follows `prereg_go_nogo.md` (v1: D = 13, and
mean-difference CAA gave no gain at any layer, so NO-GO). Same records, labels, features and
train/tune/test split (seed 0). The v1 test split was used to evaluate method M1 only; no v2 method
has touched it.

## Methods

Per-layer methods (each: unit vector × alpha × median token norm at that layer, added at the output of
block k-1 at every position; layers 1..28; alpha ∈ {0.05, 0.1, 0.2, 0.4, 0.8}):
- **M1 caa_mean**: v1 method (mean-pooled factual − hallucinated), re-run with the wider alpha grid.
- **M2 caa_last**: the same at the last prompt token.
- **M3 probe_dir**: minus the weight vector of the logistic-regression detection probe at that layer
  (standardized, mapped back to raw space), i.e. steer against the detected "will hallucinate" direction.
- **M4 contrastive_answer**: on train-split questions the model got wrong, teacher-force (a) the gold
  answer and (b) the model's own wrong answer after the prompt; vector = mean over questions of
  mean-over-answer-tokens(h_gold) − mean-over-answer-tokens(h_wrong).

Multi-layer method:
- **M5 ITI** (Li et al. 2023): per-head activations (input to o_proj) at the last answer token for the same
  gold/wrong pairs. A logistic probe per head (80/20 split of train pairs) picks the top-K heads by
  validation accuracy. Mass-mean-shift direction per head; add alpha × sigma × direction at every position.
  K ∈ {16, 48}, alpha ∈ {5, 10, 15}.

Conditional variant (for every configuration above, computed from the same generations): steer only
questions whose layer-D detector gives p(hallucinate) above tau; for the others keep the unsteered
answer. tau is set so the flagged fraction ∈ {100% (= unconditional), 50%, 30%, 20%, 10%} of questions.

## Guard against gaming the answer check

A configuration is disqualified if its mean answer length on tune is more than 2× the unsteered mean
length (listing several candidates would pass the check that a gold answer appears in the reply).

## Selection (tune split only)

Best configuration = argmax tune gain over all (method, layer, alpha / K, tau) that pass the guard.

## Decision rule (test split, evaluated once)

- **Q1: does steering work at all?** YES if the 95% paired-bootstrap CI (2000 resamples) of the best
  configuration's test gain is > 0.
- **Q2: detect layer ≠ steer layer?** Only if Q1 is YES and the best method is per-layer: evaluate that
  method at every layer (at each layer's tuned alpha / tau) on test. YES if |S − D| ≥ 4 and the CI of
  gain(S) − gain(D) > 0 (S chosen on tune, as in v1).
- Secondary (reported, not used for the decision): the best configuration of each method on test.

## Deviations

(none yet)
