# Running the current experiments

The older recovery/drift results are from their completed original runs. The
paired-evidence and response-policy studies are newly executed GPU experiments with frozen hashes
and decision timing audits. Raw generations, activation arrays, fitted models and training logs
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

## Paired evidence experiment

Use the same pinned assets and generation environment above. The data builder
reads committed prior cohort IDs for exclusions and reconstructs the corrected
v3 cohorts deterministically. Choose a fresh run root:

```bash
export ACTIVATION_RUNS="$PWD/runs/paired_reproduction"
python -m experiments.paired_evidence.data
CUDA_VISIBLE_DEVICES=0 python -m experiments.paired_evidence.collect --model qwen7b --splits pilot train validation
CUDA_VISIBLE_DEVICES=1 python -m experiments.paired_evidence.collect --model phi35 --splits pilot train validation
```

The two collection commands can run concurrently. From a clean git worktree,
run the bounded activation-only model search, then collect test with the frozen
predictors:

```bash
python -m experiments.paired_evidence.search
CUDA_VISIBLE_DEVICES=0 python -m experiments.paired_evidence.collect --model qwen7b --splits test --predictor "$ACTIVATION_RUNS/paired_evidence/qwen7b/fitted/best.joblib"
CUDA_VISIBLE_DEVICES=1 python -m experiments.paired_evidence.collect --model phi35 --splits test --predictor "$ACTIVATION_RUNS/paired_evidence/phi35/fitted/best.joblib"
python -m experiments.paired_evidence.evaluate --model qwen7b --split test
python -m experiments.paired_evidence.evaluate --model phi35 --split test
python -m experiments.paired_evidence.evaluate --model qwen7b --split pilot
python -m experiments.paired_evidence.evaluate --model phi35 --split pilot
python -m experiments.paired_evidence.report
```

The search commits trial configurations and keep/revert decisions on the current
branch; use an experiment branch. It refuses dirty or stale loaded sources.
The published search now explicitly resets and commits the mean baseline when
starting from the saved winning configuration, and handles no-op configuration
changes. This initialization fix was made after the recorded run; its original
source hashes and checkpoint are preserved in provenance.json. It does not
change the trial grid, fitted objectives or recorded results.
Twelve ridge configurations and the mean baseline are evaluated using validation
activation error only. The independent pilot informed action design and the
fixed condition map; do not describe the whole design as outcome-free.
Raw artifacts stay below the run root. Compact exports go to
`results/paired_evidence`; use a separate checkout to retain published results.
[Method and controls](experiments/paired_evidence/README.md).

## Error-risk and fixed clean-reference follow-up

Reuse the completed paired run's train/validation executions and coordinates.
Start from a clean experiment checkout. If you have just regenerated the paired
report, commit those result exports on your reproduction branch before fitting.
Model fitting refuses uncommitted source/result changes.

```bash
export PAIRED_RUN_ROOT="$ACTIVATION_RUNS"
export ACTIVATION_RUNS="$PWD/runs/error_reproduction"
python -m experiments.error_field.data --source "$PAIRED_RUN_ROOT/paired_evidence"
python -m experiments.error_field.fit
python -m experiments.error_field.reference
```

The first fit uses baseline-answer correctness and activation logit transitions;
the second adds the paired clean-state policy without changing the first policy.
Both compare 13 configurations using validation activation-derived error only.
No strategy-success labels select their learned parameters. The new 800-base test
cohort excludes all prior question IDs and source v3 candidate-pool titles for
supports/fillers. See the method for the narrower older-title exclusion scope.

If reproducing the secondary budget check, register it **before either test
collector completes**. Use a fresh output directory so the published registration
is preserved:

```bash
export RETRIEVAL_BUDGET_RESULTS="$ACTIVATION_RUNS/retrieval_budget_results"
python -m experiments.retrieval_budget.evaluate --freeze
```

The output override is a post-run packaging improvement; original registrations
and their hashes are retained unchanged. Keep this variable set while scoring
and reporting the secondary study.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.paired_evidence.collect --model qwen7b --splits test --predictor "$ACTIVATION_RUNS/paired_evidence/qwen7b/fitted/best.joblib"
CUDA_VISIBLE_DEVICES=1 python -m experiments.paired_evidence.collect --model phi35 --splits test --predictor "$ACTIVATION_RUNS/paired_evidence/phi35/fitted/best.joblib"
python -m experiments.error_field.evaluate --model qwen7b
python -m experiments.error_field.evaluate --model phi35
python -m experiments.error_field.report
python -m experiments.error_field.diagnose --model qwen7b
python -m experiments.error_field.diagnose --model phi35
python -m experiments.error_field.diagnose --report
```

The diagnostic uses actual post-action states and a separately named actual-clean
oracle. It is descriptive and cannot be used as a single-input policy. Its
additional public/confidence budget controls were added posthoc; the original
acceptance decisions are not changed. Its report derives values from JSON rather
than embedding this run's numbers.

For the secondary report, score both original exploratory and fresh cohorts:

```bash
ACTIVATION_RUNS="$PAIRED_RUN_ROOT" python -m experiments.retrieval_budget.evaluate --model qwen7b --label exploration
ACTIVATION_RUNS="$PAIRED_RUN_ROOT" python -m experiments.retrieval_budget.evaluate --model phi35 --label exploration
python -m experiments.retrieval_budget.evaluate --model qwen7b --label replication
python -m experiments.retrieval_budget.evaluate --model phi35 --label replication
python -m experiments.retrieval_budget.report
```

A fresh reconstruction may reproduce the protocol and cohort IDs without being
an independent scientific replication: the published outcomes are already known.
A new hypothesis test requires an independently reserved cohort and new record.

### Replaying the recorded local artifacts

The published branch includes a search-initialization fix and result-path override
made **after** the experiments. Source guards correctly refuse those changed
files against the original frozen artifacts. Do not rewrite provenance hashes.
For exact original analysis, use a separate checkout of the recorded sources:

- Response-policy scoring: commit `2390f595eed958fa6d65dfa849096fecebb407b1`.
- Secondary budget scoring with its registration file: commit
  `3864ba7` (the registered analysis sources are from `4ab3b57`).
- The paired run's frozen source checkpoint is recorded in
  `results/paired_evidence/provenance.json`.

Set ACTIVATION_RUNS to the existing artifact root in that checkout. Replaying
requires the original ignored generations/states/models; the committed compact
JSON files alone do not reconstruct them. Fresh runs on the current published
source freeze their own matching hashes and can use the current commands above.

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
- Completed paired-evidence and response-policy studies, activation-only search,
  secondary registration, offline diagnostics and frozen provenance.

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
