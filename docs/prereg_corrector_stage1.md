# Pre-registration: learned corrector, stage 1 (Llama-3.2-3B, TriviaQA)

Committed before any stage-1 data was generated. Stage 1 asks one thing: **can a network predict a
question-specific corrective edit from the model's own hidden state, without seeing the answer, on held-out
questions?** The pilot showed that edits *optimized with the gold answer* fix about 60% of errors at 5% of
activation norm (H2), and that no fixed vector does (H4). Abstention/gating is stage 2.

## Data

Same 3000 records, labels and train/tune/test split (seed 0) as `prereg_go_nogo.md`.
- 10 samples per question at T = 1 (all splits; used for training targets on train, and for the
  knows / doesn't-know breakdown on test).
- Training targets (train split only). The target text is always one of the model's own outputs, so no
  gold text is injected (gold is used only to decide which samples are correct):
  - **R** (recoverable errors): greedy answer wrong, at least one correct sample. Target = the most frequent
    correct sample.
  - **K** (keep): greedy answer correct. Target = the greedy answer.
  - The loss weights R and K equally in total.

## Methods (all trained on the same data, selected on tune, evaluated once on test)

- **Corrector (ours).** Reads the layer-13 residual at the last prompt token. MLP 3072→256→3072, output
  δ = ε · n₁₃ · tanh(‖u‖) · u/‖u‖, where n₁₃ is the median token norm at layer 13. δ is added at the
  output of layer 13 at every position, computed once at prefill. Loss: NLL of the target answer
  (+ EOS). AdamW, lr 1e-4, batch 16. Grid: ε ∈ {0.05, 0.1, 0.2} × epochs ∈ {1, 2, 3}.
- **Fixed vector (ablation).** The same, but δ is one learned vector shared by all questions (same
  parameterization, same grid). Tests whether input dependence matters.
- **LoRA fine-tuning (RFT baseline).** Rank 8, alpha 16, on q_proj and v_proj of every layer, same data and
  loss, lr 1e-4, epochs ∈ {1, 2, 3}.
- **No intervention.**

## Primary metric

Greedy accuracy on held-out test questions (same answer check as the labels), paired bootstrap over questions
(2000 resamples, 95% CI). Also reported: fix rate on originally wrong test questions, break rate on originally
correct ones, both split by knows (≥ 2/10 correct samples) and doesn't know (0/10).

## Decision

- **Q1, the edit is predictable:** CI of (corrector − no intervention) > 0.
- **Q2, input dependence matters:** CI of (corrector − fixed vector) > 0.
- **Q3, competitive with fine-tuning:** CI lower bound of (corrector − LoRA) > −2 pts (non-inferiority).

GO for the corrector as the paper's method if Q1 and Q2 hold. Q3 decides the pitch: if LoRA is clearly better,
the corrector is not an ICML method and the paper becomes the analysis (H2–H4) plus the self-distillation finding.

## Deviations

(none yet)
