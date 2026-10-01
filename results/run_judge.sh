#!/bin/bash
cd ~/Documents/facr-icml2027
for j in phi qwen; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 20000 ]; do sleep 60; done
  echo "$(date) start $j"; python3 scripts/judge_audit.py --judge $j > results/judge_$j.log 2>&1 && echo "$(date) done $j" || echo "$(date) FAILED $j"
done
python3 scripts/judge_audit.py --analyse > results/judge_audit.log 2>&1
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
echo "$(date) JUDGE DONE"
