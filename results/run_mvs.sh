#!/bin/bash
# Runs the mixed-vs-separated sweep over all 10 models, one at a time, detached from any editor session.
# Waits for GPU memory before each model; deletes each downloaded model (except Llama-3.2-3B) after success.
cd ~/Documents/facr-icml2027
declare -A HF=( [llama3.2-3b]=meta-llama/Llama-3.2-3B-Instruct [qwen2.5-3b]=Qwen/Qwen2.5-3B-Instruct
  [olmo2-1b]=allenai/OLMo-2-0425-1B-Instruct [granite3.1-2b]=ibm-granite/granite-3.1-2b-instruct
  [smollm2-1.7b]=HuggingFaceTB/SmolLM2-1.7B-Instruct [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct
  [qwen2.5-7b]=Qwen/Qwen2.5-7B-Instruct [olmo2-7b]=allenai/OLMo-2-1124-7B-Instruct
  [granite3.1-8b]=ibm-granite/granite-3.1-8b-instruct [mistral-7b]=mistralai/Mistral-7B-Instruct-v0.3 )
for k in llama3.2-3b smollm2-1.7b olmo2-1b granite3.1-2b qwen2.5-3b llama3.1-8b qwen2.5-7b olmo2-7b granite3.1-8b mistral-7b; do
  case $k in llama3.1-8b|qwen2.5-7b|olmo2-7b|granite3.1-8b|mistral-7b) need=21000;; *) need=11000;; esac
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt $need ]; do sleep 60; done
  echo "$(date) start $k"
  if python3 scripts/mixed_vs_separated.py --key $k > results/mixed_vs_separated_$k.log 2>&1; then
    echo "$(date) done $k"
    if [ "$k" != "llama3.2-3b" ]; then rm -rf ~/.cache/huggingface/hub/models--$(echo ${HF[$k]} | sed 's#/#--#'); fi
  else
    echo "$(date) FAILED $k"
  fi
done
python3 scripts/mixed_vs_separated.py --aggregate > results/mixed_vs_separated_aggregate.log 2>&1
echo "$(date) ALL DONE"
