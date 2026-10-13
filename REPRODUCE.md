# Running the current experiments

The committed results are from the completed original runs. The code here has
portable paths and package imports; it has not been used to claim a new GPU
replication. Raw generations, activation arrays, fitted models and training logs
are not committed. Reconstructing their numerical analyses requires collecting
those artifacts again, or obtaining the original artifacts with their original
source version. Original fitted geometry objects are not a drop-in input to this
repackaged module layout.

## Environment and tests

Use Python 3.12. The recorded dependency versions are in
[requirements-experiments.txt](requirements-experiments.txt).

```bash
python -m pip install -r requirements-experiments.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest -q
```

The unit tests run on CPU without downloading LLM weights. Full collection was
run on two H100 NVL GPUs with about 96 GB each, one model per GPU. The backend
loads both a Transformers model and a vLLM model and reserves 16 GiB for vLLM's
KV cache. A single suitably sized GPU can run the models sequentially. This
configuration has not been validated on a 24 GB card.

## Restore the exact data splits

Run from the repository root. Setup downloads pinned assets only when requested,
rebuilds each cohort in its original order, and checks its original byte-level
checksum. This includes prior-study cohorts needed for exclusion checks.

```bash
python -m experiments.prepare --download --models qwen7b phi35 --semantic
```

`--semantic` downloads the small text encoder used only by the supervised scale
study. It is unnecessary for the drift study. Omit `--models` to prepare only
the data and scoring script. Subsequent setup using cached files:

```bash
python -m experiments.prepare
```

Models/data use the normal Hugging Face cache. Optional environment overrides:

| Variable | Meaning |
|---|---|
| `HF_HUB_CACHE` | Existing Hugging Face Hub cache; set before starting Python |
| `ACTIVATION_RUNS` | Root for generated data/results; defaults to `./runs` |
| `HOTPOTQA_PATH` | Directory containing the pinned distractor Parquet files |
| `QWEN_MODEL_PATH`, `PHI_MODEL_PATH` | Existing pinned model snapshot directories |
| `SEMANTIC_MODEL_PATH` | Existing pinned MiniLM snapshot directory |

Custom snapshot paths must contain the pinned files. `--download` uses the
standard Hub cache; it does not populate arbitrary override directories.
Setup refuses to overwrite different files in a prior run. Set a new
`ACTIVATION_RUNS` directory after intentionally changing source or protocol.

## Activation-drift study

After setup, use the following environment in each terminal:

```bash
export PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
```

`PYTHONHASHSEED=0` is needed for reproducible BM25 accumulation and prompt
reconstruction. The initial preparation freezes collection and analysis sources.

For Qwen on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.activation_drift.pilot --model qwen7b --phase calibration
python -m experiments.activation_drift.supplement --model qwen7b --phase freeze
CUDA_VISIBLE_DEVICES=0 python -m experiments.activation_drift.pilot --model qwen7b --phase test
python -m experiments.activation_drift.analyze --model qwen7b
python -m experiments.activation_drift.supplement --model qwen7b --phase evaluate
python -m experiments.activation_drift.audit --model qwen7b
```

Repeat the same sequence with `phi35` in place of `qwen7b` and GPU 1 (or GPU 0
after Qwen finishes). After both models:

```bash
python -m experiments.activation_drift.analyze --report
```

Outputs appear under `runs/activation_drift`. Test action choices are saved
before candidate mitigations execute. Fitting reads only calibration/validation
states and baseline correctness. Analysis is the stage that scores mitigated
answers. Supplementary evaluations are saved separately.

## Earlier supervised recovery study

With the same environment and prepared assets:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/recovery_scale/run_model.sh qwen7b
CUDA_VISIBLE_DEVICES=1 bash experiments/recovery_scale/run_model.sh phi35
python -m experiments.recovery_scale.report
```

Each runner collects main cohorts, fits all three training sizes, evaluates,
collects two 128-question repeats, and audits. Outputs appear under
`runs/recovery_scale`. This is a substantial GPU experiment, not a smoke test.

## What was packaged

- Current drift experiment, supervised comparison, and their required helpers/tests.
- Recorded summary metrics and one figure per study.
- Cohort IDs/checksums, pinned model identities and scorer provenance.
- The proposed next test, explicitly labelled as unrun.

Source calculations, prompts, action definitions and fitting grids were retained.
Changes cover configurable paths, importable module names, restoring cohorts,
source-freeze locations and reproduction wording. The standalone presentation
finalizer from the original run was omitted because it embeds that run's test
counts and audit claims. The normal analysis reports fresh results from a new
run. Original source hashes for the included experiment files are recorded in
[results/experiment_sources.json](results/experiment_sources.json).

The existing main-branch detection/steering files remain in place. No new copies
of old notebooks, raw caches, model weights, process plans or review logs were
added by this branch.
