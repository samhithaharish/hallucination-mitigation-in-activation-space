#!/usr/bin/env bash
set -euo pipefail
model=${1:?Specify qwen7b or phi35}
case "$model" in qwen7b|phi35) ;; *) exit 2 ;; esac
python_bin=${PYTHON_BIN:-python}
export PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
study_log_dir=$("$python_bin" -c 'from experiments.paths import SCALE_RUN; print(SCALE_RUN / "logs")')
mkdir -p "$study_log_dir"
case "$model" in qwen7b) study_log=qwen-main-pipeline.log ;; phi35) study_log=phi-main-pipeline.log ;; esac
{
"$python_bin" -m experiments.recovery_scale.verified_collect --model "$model" --run-name main --splits train validation test transfer
"$python_bin" -m experiments.recovery_scale.train --model "$model" --phase fit
"$python_bin" -m experiments.recovery_scale.train --model "$model" --phase evaluate
"$python_bin" -m experiments.recovery_scale.verified_collect --model "$model" --run-name repeat1 --splits test --repeat 1 --limit 128
"$python_bin" -m experiments.recovery_scale.verified_collect --model "$model" --run-name repeat2 --splits test --repeat 2 --limit 128
"$python_bin" -m experiments.recovery_scale.audit --model "$model"
} 2>&1 | tee -a "$study_log_dir/$study_log"
