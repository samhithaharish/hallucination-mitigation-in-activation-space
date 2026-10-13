import importlib
import pytest


def test_sampling_excludes_previous_questions_and_embargoed_titles():
    try:
        d = importlib.import_module('experiments.recovery_scale.data')
    except ModuleNotFoundError:
        pytest.fail('Fresh cohort sampling is missing')
    rows = [dict(id=str(i), question=q, level='hard', type='bridge',
                 supporting_facts={'title': [t]}) for i, (q, t) in enumerate([
                     ('old question', 'A'), ('new one', 'embargo'), ('The DUP!', 'C'),
                     ('dup', 'D'), ('another', 'E'), ('last', 'F')])]
    selected = d.sample(rows, 3, {'0'}, {'old question'}, {'embargo'}, seed=17)
    assert '0' not in [r['id'] for r in selected]
    assert '1' not in [r['id'] for r in selected]
    assert len({d.normalize(r['question']) for r in selected}) == 3
    with pytest.raises(ValueError, match='eligible'):
        d.sample(rows, 7, set(), set(), set(), seed=17)
