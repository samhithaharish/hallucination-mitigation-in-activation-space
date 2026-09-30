#!/bin/bash
# Waits for the v2 run, then smoke-tests and runs the correctability pilot. Independent of any editor session.
cd ~/Documents/facr-icml2027
while kill -0 4079432 2>/dev/null; do sleep 30; done
echo "v2 finished $(date)"
python3 scripts/correctability_pilot.py --smoke > results/correctability_pilot_smoke.log 2>&1 || { echo "SMOKE FAILED $(date)"; exit 1; }
echo "smoke ok $(date)"
python3 scripts/correctability_pilot.py > results/correctability_pilot_llama3.2-3b.log 2>&1 && echo "pilot done $(date)" || echo "PILOT FAILED $(date)"
