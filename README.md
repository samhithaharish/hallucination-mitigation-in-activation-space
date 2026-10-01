# Detectable, Not Correctable: Hallucination Signals vs Internal Edits in LLMs

Target venue: **ICML 2027** (main conference). Status: experiments on TriviaQA done across 10 models; NQ-Open
replication running; writing not started.

## The claim (Path A: analysis paper)

> Language models carry an internal signal that they are about to hallucinate, but editing their internal
> activations cannot reliably fix the hallucination: a small edit reaches the model's own **plausible wrong
> answers** as easily as, or more easily than, the **true** answer. Internal edits track plausibility, not
> truth. That is why steering fails, and why "steering recovers the answer" is weak evidence that a model
> "secretly knows" it.

What does help is modest and works on the model's outputs, not its activations: when the detector fires,
choose among the model's own sampled answers, or abstain (Path B, one section of the paper).

## Paper outline and evidence so far

All tests are pre-registered (`docs/prereg_*.md`, committed before the data existed) and scored on held-out
questions with paired-bootstrap 95% CIs. Models: 5 small (Llama-3.2-3B, Qwen2.5-3B, OLMo-2-1B, Granite-3.1-2B,
SmolLM2-1.7B) and 5 large (Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B, Granite-3.1-8B, Mistral-7B-v0.3), all instruct.

| # | Finding | Evidence (TriviaQA) | Pre-registration |
|---|---|---|---|
| 1 | **Hallucination is detectable internally**, before the answer is written | Probe AUROC 0.73–0.80 across 10 models, peaking around mid-depth | `prereg_go_nogo.md`, `prereg_mixed_vs_separated.md` |
| 2 | **Fixed steering does not correct it** | 5 methods (CAA mean/last-token, probe direction, gold-vs-wrong contrastive, ITI), with and without detector gating: best +0.9 pts, CI [−0.2, +2.2] | `prereg_steering_v2.md` |
| 3 | **Corrective edits are question-specific** | Per-question edits that fix an answer are nearly orthogonal across questions (cos ≈ 0.01–0.16); their mean transfers at 4–13% of the per-question rate (all 10 models). A learned state→edit corrector gives +1.3 pts, CI [−0.2, +3.1], no better than LoRA | `prereg_correctability_pilot.md`, `prereg_corrector_stage1.md` |
| 4 | **Edits reach plausible answers, not true ones** (headline) | At 5% of activation norm, the model's own sampled wrong answer is reached as often as or more often than gold in **10/10** models (significantly more often in 7/10); unrelated answers are hard (1–12% in 8 models, higher only in the large OLMo/Granite) | `prereg_type_control.md` |
| 5 | **No scale effect from 1B to 8B** | Small vs large: detection p = 0.42, knows-fraction p = 0.21, edit transfer p = 0.66; all 10 models "mixed" | `prereg_mixed_vs_separated.md` |
| 6 | **Correct and wrong answers separate better after answering** | Answer-level probe 0.79–0.87 vs question-level 0.73–0.80, 10/10 models | `prereg_mixed_vs_separated.md` |
| 7 | **What helps (Path B):** detector-gated selection among the model's own samples | Beats greedy in 6/9 held-out models (+0.8 to +2.7 pts); higher selective accuracy at matched coverage 8–9/9; no reliability gain over calibrated abstention | `prereg_generate_then_select.md` |

## To do before submission

| Item | Status |
|---|---|
| Second dataset (NQ-Open) for findings 1, 4, 5, 6, 7 | **running** (`results/run_nq.sh`, `prereg_nq_replication.md`) |
| Audit the answer check (containment) with an LLM judge on a few hundred answers | not started |
| Same-type control at a smaller edit size for the large models (they hit the ceiling at ε = 0.05) | not started |
| Models above 8B (needs the cluster or quantization) | optional |
| One-page summary for the advisor, then a paper draft | not started |
| Related-work positioning (see `docs/related_work_2026-09-29.md`) | done; to be written up |

## Path B (method paper) and why it is not the headline

A detector-gated generate-then-select method with calibrated abstention. It works but gains only 1–3 points
against 10–18 points of headroom (a correct answer is among the samples that much more often than greedy is
right), so it supports the analysis rather than carrying the paper.

## How the project got here

The project began as "the layer that best detects a hallucination differs from the layer where steering best
corrects it". Re-running it with gold-answer scoring and held-out tests showed that steering does not correct
hallucinations at any layer (finding 2), so there is no correction layer to compare. That led to the per-question
edit analysis (finding 3), the 10-model sweep (findings 5–7) and the same-type control (finding 4), which gives
the current claim. Each step's pre-registration and result is in `docs/` and the git history.

## Layout

```
scripts/
  go_nogo.py              detection layer + mean-difference steering sweep (Llama-3.2-3B, finding 1-2)
  steering_v2.py          5 steering methods, detector-gated variants (finding 2)
  correctability_pilot.py per-question gradient edits by layer, random-answer control (finding 3)
  corrector_stage1.py     learned state->edit corrector vs fixed vector vs LoRA (finding 3)
  mixed_vs_separated.py   10-model sweep: detection, knows-fraction, edit structure, answer-level probe,
                          saves scores for Path B (--dataset triviaqa | nq_open)
  type_control.py         gold vs model-plausible wrong answers vs random (finding 4)
  path_b.py               generate-then-select with calibrated abstention (finding 7), CPU only
  clean_notebook.py       strip outputs and scan notebooks for secrets before committing
docs/
  prereg_*.md             one pre-registration per experiment, with logged deviations
  related_work_*.md       what is already published, framing by framing
  method_design.md        the learned-corrector design (tested in stage 1, not the paper's method)
  notebook_inventory.md   the original Colab notebooks and where they came from
results/                  per-experiment outputs (JSON, logs, figures); large arrays are git-ignored
notebooks/                original Colab notebooks, outputs stripped and credentials redacted
src/facr/, scripts/run_experiment.py, tests/   the earlier pipeline (token-F1 labels, fixed alpha); kept for
                          reference, superseded by the scripts above
```

## Reproduce

```bash
pip install -r requirements.txt
export HF_TOKEN=...                       # needs access to the gated Llama models

python scripts/go_nogo.py                                   # finding 1-2 base data (Llama-3.2-3B)
python scripts/steering_v2.py                               # finding 2
python scripts/mixed_vs_separated.py --key llama3.1-8b      # one model of the sweep (see MODELS in the script)
python scripts/mixed_vs_separated.py --aggregate
python scripts/type_control.py --key llama3.1-8b
python scripts/type_control.py --aggregate
python scripts/path_b.py
```

Everything runs on one 24 GB GPU (RTX 4090); 8B models need about 20 GB free. Every stage caches its
outputs, so a crashed run resumes. Layer index k means the output of transformer block k−1
(`hidden_states` convention). Labels: an answer is correct if any accepted gold answer appears in it
after normalization.

## Security

Never commit tokens. `.env` is git-ignored. Run `python scripts/clean_notebook.py <nb> --check` on any
notebook before committing it.
