import json
import numpy as np
import pytest


def test_runner_saves_report_timing_log_and_propagates_failure(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    runner = Path(__file__).resolve().parents[1] / 'run_model.sh'
    interpreter = tmp_path / 'python'
    interpreter.write_text('#!/bin/sh\n'
        'if [ "$1" = "-c" ]; then printf "%s\\n" "$STUDY_TEST_LOG_DIR"; exit 0; fi\n'
        'printf "pipeline output\\n"\n'
        'printf "pipeline error\\n" >&2\n'
        'exit 7\n')
    interpreter.chmod(0o755)
    logs = tmp_path / 'logs'
    env = {**os.environ, 'PYTHON_BIN': str(interpreter), 'STUDY_TEST_LOG_DIR': str(logs)}
    result = subprocess.run(['bash', str(runner), 'qwen7b'], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 7
    assert (logs / 'qwen-main-pipeline.log').read_text() == 'pipeline output\npipeline error\n'
from experiments.recovery_scale.actions import ACTIONS, CALLS


def test_policy_tokens_count_every_stage_and_baseline_only_once():
    from experiments.recovery_scale.report import measured_resources
    rows=[]
    for i in range(2):
        actions={a:{'input_tokens':10*(i+1)*(CALLS[j] or 1),
                    'output_tokens':2*(i+1)*(CALLS[j] or 1)} for j,a in enumerate(ACTIONS)}
        rows.append({'actions':actions})
    result=measured_resources(rows, {'mixed':np.array([0,9])})
    assert result['offline_total_input_tokens']==420
    assert result['offline_total_output_tokens']==84
    p=result['policies']['mixed']
    assert p['mean_incremental_input_tokens']==30
    assert p['mean_incremental_output_tokens']==6
    assert p['mean_total_input_tokens']==45
    assert p['mean_total_output_tokens']==9


def test_collection_timing_accumulates_retained_sessions_after_resume(tmp_path):
    from experiments.recovery_scale.report import collection_timings
    path=tmp_path/'pipeline.log'
    def row(done,seconds):
        return json.dumps({'model':'qwen7b','run':'main','split':'train','done':done,'total':192,'seconds':seconds})
    path.write_text('engine startup\n'+row(64,30)+'\n'+row(128,60)+'\n'+row(192,20)+'\n')
    result=collection_timings(path,'qwen7b')
    assert result['observed_collection_seconds']==80
    assert len(result['sessions'])==2
    assert result['sessions'][1]['seconds']==20
    assert 'excludes' in result['interpretation']
