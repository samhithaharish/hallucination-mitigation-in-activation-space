"""20-seed shuffled-label null for the T1 probe at one layer/position (prereg deviation 2026-10-20).

    python experiments/layer_scaling/shuffle_null.py <model> <layer, 1-based> <position: 0 prompt_end, 1 answer_end>
"""
import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from experiments.layer_scaling.analyze import RESULTS, fit_logit, load_split, targets

model, layer, pos = sys.argv[1], int(sys.argv[2]) - 1, int(sys.argv[3])
tr, va, te = (load_split(model, s) for s in ('train', 'validation', 'test'))
X = lambda d: d['states'][:, layer, pos].astype(np.float32)
ytr, yva, yte = (targets(d['f1'])['T1'] for d in (tr, va, te))
Xtr, Xva, Xte = X(tr), X(va), X(te)


def one(seed):
    rng = np.random.default_rng(seed)
    sc, m, _ = fit_logit(Xtr, rng.permutation(ytr), Xva, rng.permutation(yva))   # train AND validation shuffled
    return roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])


a = np.array(Parallel(n_jobs=10)(delayed(one)(s) for s in range(20)))
out = {'model': model, 'layer': layer + 1, 'position': pos, 'seeds': 20, 'aurocs': a.round(4).tolist(),
       'mean': float(a.mean()), 'sd': float(a.std()), 'min': float(a.min()), 'max': float(a.max())}
(RESULTS / f'{model}_shuffle_null.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'aurocs'}))
