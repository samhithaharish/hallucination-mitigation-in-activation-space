"""Scientific contract tests for the exploratory geometry probe."""
import numpy as np
import pytest
from experiments.activation_drift import pilot


def test_reducing_drift_selects_opposite_direction_and_keep_if_all_harmful():
    # A sign error would move the state further away; step-size matters too.
    r = np.array([[2., 0.], [-2., 0.], [0., 0.]])
    moves = np.broadcast_to(np.array([[0., 0.], [-1., 0.], [1., 0.], [-10., 0.]]), (3, 4, 2))
    scores = pilot.drift_scores(r, moves)
    assert scores.argmax(1).tolist() == [1, 2, 0]
    np.testing.assert_allclose(scores[0], [0., 1.5, -2.5, -30.])


def test_endpoints_exclude_terminal_and_capture_preceding_decision():
    # Off-by-one would expose the last answer token instead of its decision state.
    rows, points = pilot.replay_inputs([[10, 11], [12], [13, 14]], [[21, 22, 2], [23, 2], [2]], {2})
    assert rows == [[10, 11, 21], [12], [13, 14]]
    assert points == [[1, 2], [0, 0], [1, 1]]


def test_geometry_fit_has_no_action_outcome_argument():
    # A reward-dependent selector could change with action outcomes; the fit API
    # deliberately accepts only states and baseline correctness.
    rng = np.random.default_rng(33)
    p = rng.normal(size=(70, 2)); e = p + rng.normal(scale=.1, size=p.shape)
    x = np.stack([p, e], axis=1)[:, None, :, :]
    states = np.repeat(x[:, None], 3, axis=1)
    states[:, 1, :, 1, 0] -= 1.
    states[:, 2, :, 1, 0] += 1.
    ok = np.arange(70) % 2 == 0
    model = pilot.Geometry(components=2).fit(states[:50], ok[:50], states[50:], ok[50:])
    pred = model.predict(states[50:, 0])
    assert pred['operator'].shape == (20, 3)
    assert np.isfinite(pred['operator']).all()
    with pytest.raises(TypeError):
        pilot.Geometry().fit(states[:50], ok[:50], states[50:], ok[50:], action_rewards=np.zeros((50,3)))


def test_replay_padding_and_layer_time_alignment_on_real_tiny_transformer():
    # Wrong padding positions or hook axes could silently scramble drift.
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from types import SimpleNamespace
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    prompts, answers = [[10,11], [12,13,14,15]], [[20,21,22], [23,24]]
    result = pilot.capture(model, SimpleNamespace(pad_token_id=0), prompts, answers, {2}, layers=[1,2])
    assert result.shape == (2,2,2,16)
    for i,(row, points) in enumerate(zip([[10,11,20,21], [12,13,14,15,23]], [[1,3],[3,4]])):
        with torch.no_grad():
            hidden = model(torch.tensor([row]), output_hidden_states=True).hidden_states
        want = torch.stack([hidden[l][0, points] for l in (1,2)]).numpy()
        np.testing.assert_allclose(result[i], want, rtol=2e-3, atol=2e-4)
