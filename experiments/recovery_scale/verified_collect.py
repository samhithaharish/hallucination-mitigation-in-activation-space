"""Verify archived model identities before collecting or resuming frozen batches."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from .data import OUT, MODEL_PATHS, file_digest
from .collect import run as collect_run


def verify_identity(snapshot, manifest_path):
    snapshot,manifest_path=Path(snapshot),Path(manifest_path)
    expected=json.loads(manifest_path.read_text())
    # Include settings, tokenizers, weight indices and alternative/custom weights.
    relevant={str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file() and
              (p.suffix in ('.json','.safetensors','.bin','.model','.tiktoken','.py','.jinja','.jinja2')
               or p.name in ('vocab.txt','merges.txt'))}
    if relevant != set(expected):
        raise ValueError('Model identity file set changed: '+str(sorted(relevant.symmetric_difference(expected))))
    if not expected or 'config.json' not in expected:
        raise ValueError('Incomplete model identity manifest')
    for name,entry in expected.items():
        p=snapshot/name
        if p.stat().st_size!=entry['bytes'] or file_digest(p)!=entry['sha256']:
            raise ValueError('Model identity changed: '+name)
    return {'files':len(expected),'manifest_sha256':file_digest(manifest_path),
            'snapshot':str(snapshot),'verified_at_utc':datetime.now(timezone.utc).isoformat()}


def run(key,splits,run_name='main',batch_size=64,repeat=0,limit=None,out=OUT,verify_only=False):
    root=Path(out)/key
    def check(phase):
        record={**verify_identity(MODEL_PATHS[key],root/'model_hashes.json'),
                'run':run_name,'phase':phase}
        with (root/'identity_checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({'identity_verified':record}),flush=True)
    check('before')
    if not verify_only:
        collect_run(key,splits,run_name,batch_size,repeat,limit)
        check('after')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODEL_PATHS,required=True)
    p.add_argument('--splits',nargs='+',default=['calibration']);p.add_argument('--run-name',default='calibration')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--limit',type=int);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    run(a.model,a.splits,a.run_name,a.batch_size,a.repeat,a.limit,verify_only=a.verify_only)
