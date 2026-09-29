# Notebook inventory (Google Drive)

Where each paper-related Colab notebook maps to in this repo. Drive IDs are for the
Colab URL: `https://colab.research.google.com/drive/<id>`.

Status column: what I could verify from Drive metadata/contents. "unread" means I have only
seen the filename, size and dates, not the cells.

## Detection / probing (FACR, LAPSH)

| Notebook | Drive ID | Modified | Status | Maps to |
|---|---|---|---|---|
| contrastive_loss_hallucination.ipynb | 1xxTDfQltB1prcrJYQnRBRmAYXvvS8KIO | 2026-03-19 | unread | early probing (archive) |
| Untitled17.ipynb | 17zpbIHpFs-RmJ5apPedG-4XzifM6__2G | 2026-03-24 | unread, 583KB, possibly research | check |
| facr_complete.ipynb (Apr 3) | 1jRTDaDz5Fue0IsKIRYreLcdwEog2lQ8E | 2026-04-03 | unread | superseded by Apr 8? |
| facr_complete.ipynb (Apr 8) | 1LN7GuDet80PFmUK-PGUCeO3P6YhAftso | 2026-04-08 | unread | `src/facr/{features,probes}.py` |
| LAPSH_GSM8K (1).ipynb | 1me31vX0TvMNOhepCBJjCwX4xEIajzZ-W | 2026-04-15 | unread | GSM8K variant (add `data_gsm8k.py`) |
| LAPSH_GSM8K (2).ipynb | 1wKhobImtA3g30loaJuH1m6xehvLyTFVW | 2026-04-16 | unread | duplicate of (1)? |
| q4_layer_probe_comparison.ipynb | 1fUy11IoxoO0IJR0QqChKWhZrfDjcjfPg | 2026-05-05 | unread | `probes.py` comparisons |

## Steering

| Notebook | Drive ID | Modified | Status | Maps to |
|---|---|---|---|---|
| FACR_RL_Complete.ipynb | 1awW1voUoZ_TLSMGySEkoF3Zz4cZPkDtC | 2026-04-08 | unread, 889KB | `steering.py` (RL variant) |
| FACR_RL_Steering_Complete.ipynb | 1BWfQdT-Fba592-LVbBvltvVt05v8TIlO | 2026-04-09 | unread | `steering.py` |
| TriviaQA_RL_Steering.ipynb | 19FBtnTzT-HyaJen1QEKcYTrJyN4cVPFt | 2026-04-09 | unread | `steering.py` |
| TriviaQA_Boundary_Steering.ipynb | 1ZiMzomkIA1pKXQkVh_z_5CIUU_eLIlsn | 2026-04-16 | unread | `probes.select_boundary_cases` |
| steering_sweep_and_adaptive.ipynb | 1md_LnfOg_9BW9nDQFGPaDMpmhHjGsx-L | 2026-05-06 | unread | `steering.steer_layer_sweep`, adaptive alpha |

## Layer-swap ablation

| Notebook | Drive ID | Modified | Status | Maps to |
|---|---|---|---|---|
| Layer_Swap_Ablation.ipynb | 1UPDvtrlKV0WQa8k09H8lbpJxuJIo5jmv | 2026-07-04 | unread, 50KB | `ablation.py` |
| Layer_Swap_Ablation (1).ipynb | 1bAg5AijaWLPBncYKCxKzq0mWrN2dnFDm | 2026-07-04 | unread, 498KB (likely has outputs) | **read this for the 3B results** |

## Scaling

| Notebook | Drive ID | Modified | Status | Maps to |
|---|---|---|---|---|
| FACR_Scale_LLaMA3.1-8B.ipynb | 1qRu8mq3-y8gTXMkLa508yX0uITTTjQrv | 2026-07-21 | **read in full** | whole `src/facr/` + `scripts/run_experiment.py` |

Findings from the 8B notebook:
- **The run did not finish.** Cell 3 (model load) failed with a torch/torchaudio CUDA-version mismatch
  (PyTorch built for CUDA 13.0, TorchAudio for 12.8) right after a `pip uninstall -y torchvision`
  cell. Cells 4 to 12 are marked "aborted" with no outputs, so there is no 8B result yet.
- Two Hugging Face tokens were hard-coded in cells. They must be revoked; this repo reads `HF_TOKEN`
  from the environment.
- The steering-accuracy function scored "correct" as "steered output differs from the original
  generation" (the notebook's own placeholder comment says so). `steering.eval_steering_accuracy`
  now scores against gold aliases instead.
- The cross-model table has Qwen-2.5-3B and Mistral-3B rows set to `None`; fill them from the 3B runs.

## Not paper-related (leave out of this repo)

SILC / ASR (whisperx, pyannote, transcription_*, linguistics_pipeline_*, asr_pipeline,
whisper_lora_transcribe, SILC_*), coursework (CSE572-*, EEE515-*, homework0, Lab1), and empty
`Untitled*` notebooks.

## Unsure

`parametric_memory_*` (Apr 25 to 27, up to 1.2MB) may be related to the knowledge-localization
direction; check before deciding.

## In this repo

Each notebook above is committed as `notebooks/NN_<name>.clean.ipynb` (outputs, Colab metadata and widget
state stripped; hard-coded tokens replaced with `<REDACTED>`), and its cell outputs are in
`results/notebook_outputs/NN_<name>.txt`. `02_untitled17` is included pending a check that it is FACR work.
The raw notebooks are in `notebooks/archive/` locally and git-ignored.
