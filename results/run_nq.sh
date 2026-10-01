#!/bin/bash
# NQ-Open replication (docs/prereg_nq_replication.md): for each model, the sweep then the type control,
# then delete the download. Waits for GPU memory and retries up to 3 times.
cd ~/Documents/facr-icml2027
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A HF=( [llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct [qwen2.5-3b]=Qwen/Qwen2.5-3B-Instruct
  [olmo2-1b]=allenai/OLMo-2-0425-1B-Instruct [granite3.1-2b]=ibm-granite/granite-3.1-2b-instruct
  [smollm2-1.7b]=HuggingFaceTB/SmolLM2-1.7B-Instruct [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct
  [qwen2.5-7b]=Qwen/Qwen2.5-7B-Instruct [olmo2-7b]=allenai/OLMo-2-1124-7B-Instruct
  [granite3.1-8b]=ibm-granite/granite-3.1-8b-instruct [mistral-7b]=mistralai/Mistral-7B-Instruct-v0.3 )
for k in llama3.2-3b smollm2-1.7b olmo2-1b granite3.1-2b qwen2.5-3b llama3.1-8b qwen2.5-7b olmo2-7b granite3.1-8b mistral-7b; do
  case $k in llama3.1-8b|qwen2.5-7b|olmo2-7b|granite3.1-8b|mistral-7b) need=21000;; *) need=11000;; esac
  for attempt in 1 2 3; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt $need ]; do sleep 60; done
    echo "$(date) start $k (attempt $attempt)"
    if python3 scripts/mixed_vs_separated.py --key $k --dataset nq_open > results/nq_mvs_$k.log 2>&1; then
      if grep -q excluded results/nq/mixed_vs_separated/$k/summary.json || \
         python3 scripts/type_control.py --key $k --src results/nq/mixed_vs_separated --out results/nq/type_control > results/nq_tc_$k.log 2>&1; then
        echo "$(date) done $k"
        [ "$k" != "llama3.2-3b" ] && rm -rf ~/.cache/huggingface/hub/models--$(echo ${HF[$k]} | sed 's#/#--#')
        break
      fi
    fi
    echo "$(date) FAILED $k (attempt $attempt)"; sleep 300
  done
done
python3 scripts/mixed_vs_separated.py --aggregate --dataset nq_open > results/nq_mvs_aggregate.log 2>&1
python3 scripts/type_control.py --aggregate --out results/nq/type_control > results/nq_tc_aggregate.log 2>&1
MVS_ROOT=results/nq/mixed_vs_separated python3 scripts/path_b.py > results/nq_path_b.log 2>&1
echo "$(date) NQ DONE"
