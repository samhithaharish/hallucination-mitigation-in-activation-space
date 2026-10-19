# Pre-registration: layer-wise linear separability of "which fix to apply"

Written 2026-10-19, before any data below exists. Extends Som Sagar's `recovery_scale` study
(branch `activation-repair-value`) without changing its prompts, actions or scoring.

## Questions

1. **Separability by depth.** At each layer, is a linear readout of the model's activations able to tell
   (a) whether the initial answer can be fixed, (b) whether each specific fix helps, (c) which fix is best?
2. **Usefulness.** Does an activation-based fix selector beat a **confidence-based** selector at the same
   extra-generation budget? (Som's results and the advisor's question both point to confidence as the bar.)
3. **Scale.** Do 1 and 2 hold across model families and sizes?

## Data and models

- Som's frozen HotpotQA distractor cohorts (checksums verified by `experiments.prepare`): `train` first 6,000,
  `validation` 1,000, `test` 2,000; initial evidence k = 2. Same ten actions: keep, retry, retrieve, reason,
  both, fresh, prune, query, verify, vote (calls 0,1,1,1,1,1,1,2,2,3).
- Stage 1 (one RTX 4090): Phi-3.5-mini (Som's pinned revision), Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct.
- Stage 2 (if disk/cluster allow): Qwen2.5-7B-Instruct (Som's pinned revision), Llama-3.1-8B, OLMo-2-7B.
- States: every block output (`ACTIVATION_LAYERS=all`), last position, at the prompt end and at the last
  initial-answer token. Engine: Transformers greedy (vote: seeded sampling, as in Som's code).

## Targets (from answer-token F1 of each action, as Som scores it)

- **T1 fixable:** max over the nine non-keep actions of F1 > F1(keep).
- **T2 helps(a):** F1(a) > F1(keep), for each of the nine actions.
- **T3 best:** the action with the highest F1, ties broken toward fewer calls (keep wins ties).

## Analysis (frozen)

- Per layer and position: standardized features, logistic regression (T1, T2) or multinomial (T3),
  L2 strength chosen on `validation` from {1e-4, 1e-3, 1e-2, 1e-1, 1}.
- Metrics on `test`: AUROC (T1, T2), accuracy and mean regret in F1 (T3).
- **Primary layer** for each model/target: the layer and position with the best `validation` score. Only that
  layer is tested; the full depth curves are reported descriptively.
- Baselines on the same splits: (i) Som's 13 confidence features; (ii) confidence + Som's metadata features;
  (iii) shuffled-label control at the primary layer (expected AUROC 0.5); (iv) a one-hidden-layer MLP on the
  primary layer, to separate "linear" from "nonlinear" information.
- **Selector:** for each action, ridge regression of F1 gain over keep from the primary layer (T1's layer);
  choose argmax(predicted gain − λ·calls), with λ set on `validation` so mean extra calls ≤ 1.0. The
  confidence selector is built the same way from baseline (i). Compare test F1 with a paired bootstrap
  (4,000 resamples over questions). Also reported: keep, best single fixed action at matched budget, oracle.

## Hypotheses and pass criteria

- **H1 (fixability is linearly readable beyond confidence):** primary-layer T1 AUROC minus confidence AUROC,
  paired 95% CI > 0, in at least 2 of 3 stage-1 models.
- **H2 (fix-specific information):** for at least 3 of the 9 actions, T2 AUROC beats confidence (CI > 0)
  in at least 2 of 3 models.
- **H3 (useful selector):** activation selector minus confidence selector test F1, paired 95% CI > 0, in at
  least 2 of 3 models.
- Descriptive: depth of the best layer; MLP minus linear gap (a gap with CI > 0 means the information is
  present but not linearly separable).

A failed hypothesis is reported as a result, not reworded. Deviations are logged below with dates.

## Deviations

(none yet)
