# Pre-registration: detect-layer vs steer-layer go/no-go (Llama-3.2-3B, TriviaQA)

Written and committed **before** any data for this run was generated. The analysis in
`scripts/go_nogo.py` implements exactly this; any later deviation gets listed at the bottom with a
reason.

## Claim under test

The layer where a linear probe best *detects* that the model will hallucinate (D) is not the layer
where adding a steering vector best *corrects* hallucinations (S). Specifically, steering at S fixes
answers significantly better than steering at D.

## Data and labels

- Model: `meta-llama/Llama-3.2-3B-Instruct`, bf16, greedy decoding, max 24 new tokens, chat template,
  instruction "Answer with only the answer, no explanation."
- Questions: first 3000 of TriviaQA `rc.nocontext` validation.
- Label: correct (factual, 0) if any normalized gold alias appears as a whole-word span in the
  normalized answer (lowercase, punctuation and articles removed); otherwise hallucinated (1).
- Sanity gate: if the factual rate is outside 25% to 75%, stop and fix the labelling before any
  steering run.
- Split (seed 0, stratified): train 70% (2100), tune 15% (450), test 15% (450).

## Detection (D)

- Features: residual stream at every layer (hidden_states index 0..28, where k = output of block k-1),
  mean-pooled over prompt tokens excluding BOS. Secondary (reported, not used for D): last prompt token.
- Probe: StandardScaler + logistic regression (C = 0.05), 5-fold stratified CV repeated 4 times,
  **on the train split only**.
- D = layer with the highest mean CV AUROC over layers 1..28. Uncertainty: argmax per repeat, and
  per-layer mean ± std across the 20 folds.

## Steering (S)

- Vector per layer: mean(factual) − mean(hallucinated) of the train-split features, unit-normalized,
  scaled by alpha × median per-token residual norm at that layer (so alpha is a relative strength and
  comparable across layers). Added at every position at the output of block k-1.
- Alpha grid: {0.05, 0.1, 0.2, 0.4}.
- Metric per question: correct/incorrect after steering (same rule as the labels). Gain = steered
  accuracy − unsteered accuracy on the same questions, so breaking correct answers counts against a layer.
- Tune split: for each layer 1..28, choose the alpha with the best gain; S = layer with the best tuned gain.
- Test split (untouched until then): evaluate every layer at its tuned alpha.

## Decision rule (all on the test split)

GO (the claim holds on this model) only if all three hold:
1. Steering at S helps: 95% paired-bootstrap CI (2000 resamples over questions) of gain(S) > 0.
2. The layers are far apart: |S − D| ≥ 4 layers (≈14% of depth).
3. The layer choice matters: 95% paired-bootstrap CI of gain(S) − gain(D) > 0.

If 1 fails, steering does not work at any layer under honest scoring: NO-GO for the steering claim, and the
paper pivots to the evaluation framing. If 1 holds but 2 or 3 fails, the detect and steer layers are not
distinguishable: NO-GO for "different layers".

## Deviations

(none yet)
