"""Local paths and the exact dataset/model revisions used in the experiments."""
import os
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(HF_HUB_CACHE)
RUNS = Path(os.environ.get('ACTIVATION_RUNS', REPO / 'runs')).expanduser().resolve()
PILOT_RUN = RUNS / 'recovery_pilot'
SCALE_RUN = RUNS / 'recovery_scale'
DRIFT_RUN = RUNS / 'activation_drift'

DATA_REPO = 'hotpotqa/hotpot_qa'
DATA_REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
MODEL_SPECS = {
    'qwen7b': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
    'phi35': ('microsoft/Phi-3.5-mini-instruct', '2fe192450127e6a83f7441aef6e3ca586c338b77'),
}
SEMANTIC_SPEC = ('sentence-transformers/all-MiniLM-L6-v2',
                 '1110a243fdf4706b3f48f1d95db1a4f5529b4d41')


def snapshot(repo_id, revision, kind='models'):
    return CACHE / (kind + '--' + repo_id.replace('/', '--')) / 'snapshots' / revision


CORPUS = Path(os.environ.get('HOTPOTQA_PATH',
    snapshot(DATA_REPO, DATA_REVISION, 'datasets') / 'distractor')).expanduser().resolve()
MODEL_PATHS = {
    key: str(Path(os.environ.get(variable, snapshot(*MODEL_SPECS[key]))).expanduser().resolve())
    for key, variable in [('qwen7b', 'QWEN_MODEL_PATH'), ('phi35', 'PHI_MODEL_PATH')]
}
SEMANTIC_PATH = str(Path(os.environ.get('SEMANTIC_MODEL_PATH',
    snapshot(*SEMANTIC_SPEC))).expanduser().resolve())
