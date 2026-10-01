#!/bin/bash
# Smaller edit budget for the 5 large models, both datasets (docs/prereg_small_eps.md).
cd ~/Documents/facr-icml2027
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A HF=( [llama3.1-8b]=meta-llama/Llama-3.1-8B-Instruct [qwen2.5-7b]=Qwen/Qwen2.5-7B-Instruct
  [olmo2-7b]=allenai/OLMo-2-1124-7B-Instruct [granite3.1-8b]=ibm-granite/granite-3.1-8b-instruct
  [mistral-7b]=mistralai/Mistral-7B-Instruct-v0.3 )
declare -A SRC=( [triviaqa]=results/mixed_vs_separated [nq_open]=results/nq/mixed_vs_separated )
for k in llama3.1-8b qwen2.5-7b olmo2-7b granite3.1-8b mistral-7b; do
  for ds in triviaqa nq_open; do
    [ -f results/small_eps/$ds/$k/type_control.json ] && continue
    for attempt in 1 2 3; do
      while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 21000 ]; do sleep 60; done
      echo "$(date) start $k $ds (attempt $attempt)"
      if python3 scripts/type_control.py --key $k --src ${SRC[$ds]} --out results/small_eps/$ds \
           --eps 0.01,0.02,0.03 --skip prompted_wrong > results/small_eps_${k}_$ds.log 2>&1; then
        echo "$(date) done $k $ds"; break
      fi
      echo "$(date) FAILED $k $ds"; sleep 300
    done
  done
  rm -rf ~/.cache/huggingface/hub/models--$(echo ${HF[$k]} | sed 's#/#--#')
done
python3 scripts/small_eps_analysis.py > results/small_eps_analysis.log 2>&1
echo "$(date) SMALL EPS DONE"
