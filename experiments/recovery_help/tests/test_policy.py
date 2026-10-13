import numpy as np


def test_allocator_obeys_budget_and_uses_only_predicted_gains():
    from experiments.recovery_help.policy import allocate
    gains = np.array([[0., .8, .1], [0., .1, .9], [0., -.1, -.2]])
    costs = np.array([[0., 2., 2.], [0., 2., 2.], [0., 2., 2.]])
    choice, multiplier = allocate(gains, costs, 4.)
    assert choice.tolist() == [1, 2, 0]
    assert costs[np.arange(3), choice].sum() <= 4
    assert multiplier >= 0
    assert allocate(gains, costs, 0.)[0].tolist() == [0, 0, 0]


def test_allocator_general_costs_and_dual_bound():
    from experiments.recovery_help.policy import allocate, dual_upper_bound
    rng = np.random.default_rng(32)
    gains = np.column_stack([np.zeros(100), rng.normal(size=(100, 4))])
    costs = np.column_stack([np.zeros(100), rng.uniform(1, 9, (100, 4))])
    choice, multiplier = allocate(gains, costs, 60.)
    value = gains[np.arange(100), choice].sum()
    assert costs[np.arange(100), choice].sum() <= 60. + 1e-8
    assert value <= dual_upper_bound(gains, costs, 60., multiplier) + 1e-8
