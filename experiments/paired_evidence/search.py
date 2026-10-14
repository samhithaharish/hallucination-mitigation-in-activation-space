"""Bounded classic autoresearch loop; its metric contains no answer scores.

One shared configuration is selected by mean validation activation NMSE across
the two backbones. Each candidate changes only predictor.json, is committed,
verified and guarded, then kept or reverted. Run in a clean worktree.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.paths import REPO
from experiments.recovery_help.data import dump, file_digest
from .data import OUT

MODELS = ('qwen7b', 'phi35')
CONFIG = Path(__file__).with_name('predictor.json')


def command(args):
    return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()


def verify(trial):
    metrics = {}
    for key in MODELS:
        value = command([sys.executable, '-m', 'experiments.paired_evidence.fit',
                         '--model', key, '--trial', trial, '--metric-only'])
        metrics[key] = float(value)
        if not math.isfinite(metrics[key]) or metrics[key] < 0:
            raise ValueError('Invalid activation NMSE: '+value)
    return sum(metrics.values()) / len(metrics), metrics


def search(wait=False):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0')
    if wait:
        deadline = time.monotonic() + 7200
        while not all((OUT/k/'validation/complete.json').exists() for k in MODELS):
            if time.monotonic() > deadline:
                raise TimeoutError('Development collection did not finish')
            time.sleep(30)
    if command(['git', 'status', '--porcelain']):
        raise ValueError('Commit existing changes before starting the loop')
    guard = [sys.executable, '-m', 'pytest', '-q']
    command(guard)
    directory = REPO/'autoresearch/loop-261013-paired'
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory/'results.tsv'
    if ledger.exists():
        raise ValueError('Preserve the existing experiment ledger')
    ledger.write_text('# metric_direction: lower_is_better\n'
                      'iteration\ttimestamp\tcommit\tmetric\tdelta\tguard\tguard-metric\tstatus\tdescription\n')
    details = []

    def record(i, commit, metric, delta, status, description, per_model, guard_status='pass'):
        now = datetime.now(timezone.utc).isoformat()
        metric_text = 'nan' if metric is None else f'{metric:.9g}'
        delta_text = 'nan' if delta is None else f'{delta:.9g}'
        with ledger.open('a') as stream:
            stream.write(f'{i}\t{now}\t{commit}\t{metric_text}\t{delta_text}\t{guard_status}\t-\t{status}\t{description}\n')
        details.append(dict(iteration=i, timestamp=now, commit=commit, metric=metric,
                            status=status, description=description, per_model=per_model,
                            guard=guard_status))
        print(json.dumps(details[-1]), flush=True)
        dump(directory/'trials.json', details)

    initial, per_model = verify('trial_00')
    best, best_trial = initial, 'trial_00'
    record(0, command(['git', 'rev-parse', 'HEAD']), initial, 0., 'baseline',
           'mean displacement', per_model)
    candidates = [dict(kind='ridge', alpha=alpha, features=features)
                  for features in ('state', 'z')
                  for alpha in (.01, .1, 1., 10., 100., 1000.)]
    for i, config in enumerate(candidates, 1):
        description = f"{config['features']} ridge alpha={config['alpha']}"
        CONFIG.write_text(json.dumps(config, sort_keys=True)+'\n')
        command(['git', 'add', str(CONFIG.relative_to(REPO))])
        command(['git', 'commit', '-m', 'experiment: '+description])
        commit = command(['git', 'rev-parse', 'HEAD'])
        trial = f'trial_{i:02d}'
        per_model = {}
        try:
            metric, per_model = verify(trial)
            command(guard)
        except Exception as error:
            command(['git', 'revert', 'HEAD', '--no-edit'])
            (directory/f'{trial}_error.txt').write_text(str(error)+'\n'+getattr(error, 'output', ''))
            record(i, commit, None, None, 'crash', description,
                   per_model, guard_status='error')
            continue
        delta = metric-best
        if metric < best:
            best, best_trial = metric, trial
            status = 'keep'
        else:
            command(['git', 'revert', 'HEAD', '--no-edit'])
            status = 'discard'
        record(i, commit, metric, delta, status, description, per_model)
    frozen = {}
    for key in MODELS:
        source = OUT/key/'fitted'/f'{best_trial}.joblib'
        target = source.with_name('best.joblib')
        if target.exists():
            raise ValueError('A frozen predictor already exists')
        shutil.copy2(source, target)
        meta = json.loads(source.with_suffix('.json').read_text())
        meta.update(frozen_time_unix=time.time(), selected_trial=best_trial,
                    shared_validation_activation_nmse=best,
                    selection_metric='mean validation activation NMSE across both backbones',
                    checkpoint=command(['git', 'rev-parse', 'HEAD']),
                    source_hashes={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')})
        dump(target.with_suffix('.json'), meta)
        frozen[key] = meta
    summary = dict(initial_metric=initial, final_metric=best, selected_trial=best_trial,
                   kept=sum(r['status']=='keep' for r in details), trials=details, frozen=frozen,
                   action_success_labels_used=False)
    dump(directory/'summary.json', summary)
    dump(directory/'handoff.json', dict(version='2.1.0', source='loop', status='BOUNDED',
         results_tsv=str(ledger), config=dict(scope=str(CONFIG), direction='lower_is_better',
         metric='validation activation NMSE', iterations=12), findings=[f'Selected {best_trial}']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('trials','frozen')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    search(args.wait)
