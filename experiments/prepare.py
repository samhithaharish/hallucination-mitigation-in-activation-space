"""Restore the recorded cohorts and freeze sources for a fresh local run."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from . import paths


def checksum(content):
    return hashlib.sha256(content).hexdigest()


def write_checked(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite {path}; choose a fresh ACTIVATION_RUNS directory')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def restore_cohorts(manifest_path, destination, rows_by_id):
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get('split_hashes', manifest.get('hashes'))
    payloads = {}
    for split, ids in manifest['ids'].items():
        rows = [rows_by_id[identifier] for identifier in ids]
        content = json.dumps(rows, indent=2, allow_nan=False).encode()
        if checksum(content) != expected[split]:
            raise ValueError(f'Frozen cohort checksum differs: {split}')
        payloads[split] = content
    for split, content in payloads.items():
        write_checked(destination / 'data' / f'{split}.json', content)
    write_checked(destination / 'data_manifest.json', manifest_path.read_bytes())


def download_assets(models, semantic=False):
    from huggingface_hub import snapshot_download
    snapshot_download(paths.DATA_REPO, repo_type='dataset', revision=paths.DATA_REVISION,
                      allow_patterns=['distractor/*.parquet'])
    for key in models:
        spec = paths.MODEL_SPECS[key]
        manifest = json.loads((paths.REPO / 'results/recovery_scale' / f'{key}_model.json').read_text())
        snapshot_download(spec[0], revision=spec[1], allow_patterns=list(manifest))
    if semantic:
        snapshot_download(paths.SEMANTIC_SPEC[0], revision=paths.SEMANTIC_SPEC[1],
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '1_Pooling/*'])


def prepare(download=False, models=(), semantic=False):
    if download:
        download_assets(models, semantic)
    import pyarrow.parquet as pq
    from .recovery_help.data import file_digest

    experiments = [('recovery_pilot', paths.PILOT_RUN),
                   ('recovery_scale', paths.SCALE_RUN), ('activation_drift', paths.DRIFT_RUN)]
    manifests = {name: paths.REPO / 'results' / name / 'cohorts.json' for name, _ in experiments}
    source_hashes = {}
    for manifest in manifests.values():
        for name, sha in json.loads(manifest.read_text())['source_hashes'].items():
            if name in source_hashes and source_hashes[name] != sha:
                raise ValueError('Experiments disagree about dataset source: ' + name)
            source_hashes[name] = sha
    rows = {}
    for name, sha in sorted(source_hashes.items()):
        source = paths.CORPUS / name
        if not source.exists():
            raise FileNotFoundError(f'{source}: run with --download or set HOTPOTQA_PATH')
        if file_digest(source) != sha:
            raise ValueError('Dataset source checksum differs: ' + name)
        for row in pq.read_table(source).to_pylist():
            rows[row['id']] = row
    for name, destination in experiments:
        restore_cohorts(manifests[name], destination, rows)

    for key in paths.MODEL_SPECS:
        original = paths.REPO / 'results/recovery_scale' / f'{key}_model.json'
        write_checked(paths.SCALE_RUN / key / 'model_hashes.json', original.read_bytes())

    scorer_metadata = paths.REPO / 'results/recovery_pilot/scorer.json'
    scorer = paths.PILOT_RUN / 'audit_sources/hotpot_evaluate_v1.py'
    metadata = json.loads(scorer_metadata.read_text())
    if scorer.exists():
        source = scorer.read_bytes()
    elif download:
        with urlopen(metadata['url'], timeout=60) as response:
            source = response.read()
    else:
        raise FileNotFoundError(f'{scorer}: use --download once to fetch the official scorer')
    if checksum(source) != metadata['sha256']:
        raise ValueError('Official scorer checksum differs')
    write_checked(scorer, source)
    write_checked(scorer.with_suffix('.source.json'), scorer_metadata.read_bytes())

    code = paths.REPO / 'experiments/activation_drift'
    sources = sorted((paths.REPO / 'experiments').rglob('*.py')) + [code / 'README.md']
    collection = {'files': {str(p.relative_to(paths.REPO)): file_digest(p) for p in sources}}
    analysis = {'files': {name: file_digest(code / name) for name in ['analyze.py', 'supplement.py']}}
    for filename, record in [('source_manifest.json', collection), ('analysis_frozen.json', analysis)]:
        write_checked(paths.DRIFT_RUN / filename, json.dumps(record, indent=2).encode())
    print(json.dumps({'runs': str(paths.RUNS), 'cohorts_restored': list(manifests),
                      'original_split_checksums_match': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch pinned data and the official scorer')
    parser.add_argument('--models', nargs='*', choices=paths.MODEL_SPECS, default=[],
                        help='With --download, also fetch these frozen model snapshots')
    parser.add_argument('--semantic', action='store_true', help='Also fetch the scale study text encoder')
    args = parser.parse_args()
    prepare(args.download, args.models, args.semantic)
