"""Load the checksum-pinned official HotpotQA answer scorer."""
import ast
import json
import re
import string
from collections import Counter
from .data import OUT, file_digest


def official_scorer():
    source = OUT/'audit_sources/hotpot_evaluate_v1.py'
    provenance = json.loads(source.with_suffix('.source.json').read_text())
    assert file_digest(source) == provenance['sha256']
    tree = ast.parse(source.read_text())
    names = {'normalize_answer','f1_score','exact_match_score'}
    # Execute only the three audited pure metric functions, without main/import
    # side effects from the downloaded original evaluation program.
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'re': re, 'string': string, 'Counter': Counter}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['f1_score'], namespace['exact_match_score']
