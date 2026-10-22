# Pre-registration: choosing the evidence with the probe (equal-size contexts)

Written 2026-10-22, before any data below exists. Pilot model: Llama-3.2-3B-Instruct.

## Why

In the input-drift study, a probe-picked input tied "always add more evidence", partly because candidates of
different length shift the probe. Here every candidate has the same size, so only the content differs.
Close prior work: SeaKR (arXiv 2406.19215) reranks passages by reduction in internal uncertainty (needs several
samples); Jin et al. (arXiv 2509.06472) train a probe on pre-generation mid-layer states to score passages and
distil it into a reranker. This pilot is therefore a comparison, not a new-method claim.

## Data

- The first 1,000 questions of Som's frozen `test` cohort (same as the drift study), HotpotQA distractor setting
  (10 paragraphs per question).
- Candidates: every pair of the question's paragraphs (45 for 10 paragraphs), each shown with Som's keep prompt
  (order within the pair by BM25, as in Som's code). Greedy answer, cap 64 tokens.
- States at the prompt end and last answer token, at layers {axis-6, axis-3, axis, axis+3, axis+6}, where
  axis = the drift-study probe layer (18 for Llama). Answers graded by the Qwen2.5-7B judge (FACR prompt).

## Rankers (each picks one pair per question)

1. **Probe, zero-shot:** the drift-study correct/wrong probe (trained on the 6,000 baseline `train` states, no
   pair labels), prompt-end logit at the axis layer. Needs one forward pass per candidate, no generation.
2. BM25 top-2 (Som's baseline prompt).
3. Random pair (seeded).
4. Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), top-2 paragraphs by score.
5. Confidence: highest mean token log-prob of the generated answer (needs a generation per candidate).
6. **Probe, pair-trained:** logistic regression on candidate states from questions 1-500 with judged labels,
   layer and C chosen by 5-fold CV on those questions; evaluated on questions 501-1000.

Reference points: the gold supporting pair, and the oracle (any candidate judged correct).

## Hypotheses (judged accuracy of the picked pair, questions 501-1000, paired bootstrap 95% CI)

- **C1:** probe zero-shot minus BM25 top-2 > 0.
- **C2:** probe zero-shot minus cross-encoder: reported (prior work suggests rerankers are strong).
- **C3:** probe pair-trained minus probe zero-shot > 0.
- Descriptive: each ranker against confidence (which costs 45 generations per question).

Failed hypotheses are reported as results. Deviations are logged below with dates.

## Deviations

- 2026-10-22 (before any data was analysed): ranker 6 uses prompt-end states only, so that, like ranker 1, it
  needs no generation. The candidate-level AUROC of ranker 1 on the test questions is also reported.
