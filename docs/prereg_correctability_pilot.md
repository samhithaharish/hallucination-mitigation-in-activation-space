# Pre-registration: correctability-map pilot (Llama-3.2-3B, TriviaQA)

Committed before any pilot data was generated. This is an exploratory pilot on the **train** split
(tune/test stay reserved for a confirmation run). Its job is to decide whether the correctability
framing is worth a full study.

## Question

For a question the model gets wrong, what is the smallest activation edit at layer ℓ that makes greedy
decoding produce the gold answer, and how does that cost vary with layer, compared with where
hallucination is most detectable (D = 13 from `prereg_go_nogo.md`)?

## Setup

- 120 hallucinated questions sampled (seed 1) from the train split of the go/no-go records.
- Knowledge status: 10 samples at temperature 1.0 with no edit. **Knows (HK+)** if ≥ 2/10 samples are
  correct, **doesn't know (HK−)** if 0/10, and excluded from the HK comparison if exactly 1/10.
- Layers ℓ ∈ {2, 4, 8, 10, 13, 16, 20, 24, 27}.
- Edit: one vector δ per question added to the output of block ℓ-1 at every position. δ is optimized
  with Adam for 30 steps to minimize the teacher-forced NLL of the target answer, projected onto
  ‖δ‖ ≤ ε × (median token norm at ℓ). Budgets ε ∈ {0.05, 0.1, 0.2, 0.4, 0.8}.
- Success: greedy decoding with δ applied gives the target (same containment check as the labels).
  Minimal cost = smallest ε that succeeds (∞ if none does).
- Targets: **gold** (the question's gold answer) and **control** (the gold answer of a different random
  question). Forcing the control answer measures the cost of arbitrarily writing an answer at that layer.
- Transfer: at ε ∈ {0.2, 0.4}, the mean of the per-question gold δs from one half of the questions
  (renormalized to the budget) is applied to the other half, and vice versa.

## Hypotheses and how each is read

- **H1 dissociation.** The layer with the highest gold fix rate at ε = 0.2 (S*) differs from D by ≥ 4
  layers, and the paired-bootstrap CI (2000 resamples over questions) of fixrate(S*) − fixrate(D) > 0.
- **H2 correction, not overwriting.** At S*, the gold fix rate exceeds the control fix rate
  (paired-bootstrap CI > 0). If gold ≈ control, edits at S* are output forcing, and H1 does not count
  as correction.
- **H3 knowledge.** At S*, the minimal cost is lower for HK+ than for HK− (Mann-Whitney, p < 0.05).
- **H4 why fixed steering fails.** The mean pairwise cosine of per-question gold δs is < 0.2 at every layer,
  and the transferred mean δ fixes < 1/3 as many questions as the per-question δs at the same ε.

Go for the full study if H1 and H2 both hold. H3 and H4 are the mechanism.

## Deviations

(none yet)
