# Results collected from Google Drive (2026-09-29)

Everything here was produced by the earlier Colab runs, before the fixes described in the top-level
README (gold-alias steering metric, normalized vectors, held-out boundary cases). Treat these numbers as
provenance and a sanity reference, not as paper numbers; rerun with `scripts/run_experiment.py`.

| Path | Drive source | Contents |
|---|---|---|
| `drive/facr_checkpoints/` | `facr_checkpoints/` (Mar 18) | first contrastive-LoRA run: labeled data, activation PCA, trainer states |
| `drive/facr_v2/` | `facr_v2/` (Mar 24) | v2 run: layer AUROC sweep, step1 (layer curves, transfer to NQ/TruthfulQA/HaluEval), step2 (PCA, early detection) |
| `drive/facr_triviaqa/` | `facr_triviaqa/` (Apr 9) | TriviaQA RL steering: layer sweep, early detection, eval results, per-checkpoint labels, figures 1-5 |
| `drive/q4_layer_comparison/` | `q4_layer_comparison/` (May 5) | layer-probe comparison figure |
| `notebook_outputs/NN_*.txt` | Colab notebooks | text outputs of every executed cell, extracted before the notebooks were stripped (credentials redacted) |

Not in git (too large or binary, still on Drive): LoRA adapters and optimizer states
(`*/checkpoint-*/`, `*/final_model/`), `steering_vector.pt`, `steering_*_best.pt`,
`step2_results/proj_head_layer13.pt`, `q4_layer_comparison/{hidden_states_cache.pt,layer_aurocs.npy}`,
`steering_sweep/{steering_sweep_results.pt,probe_l13.pt}`.
