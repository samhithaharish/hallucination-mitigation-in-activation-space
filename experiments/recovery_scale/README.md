# Recovery selection with outcome supervision

This is the earlier large comparison: learn expected answer-quality gains for
ten recovery actions from one initial model execution. It uses action-success
labels during training. The later [activation-drift experiment](../activation_drift)
removes that particular supervision.

Six thousand training, 1,000 validation, 2,000 test and 1,000 shifted test
questions per frozen backbone; the shifted split reduces initial evidence and
uses HotpotQA validation questions. Models and action definitions are shared
with the drift study. The main comparison uses one additional generation per
question on average. Selection is frozen using validation data.

Read [RESULTS.md](../../RESULTS.md) and the
[recorded summary](../../results/recovery_scale/summary.json).
[REPRODUCE.md](../../REPRODUCE.md) gives setup and commands.

`actions.py` defines the prompts and costs; `backend.py` and `collect.py` gather
states and action outcomes; `models.py` and `train.py` fit/select/evaluate;
`verified_collect.py` and `audit.py` check identities and reconstructed results.
`report.py` produces detailed reports after collection, fitting, repeats and
audits. `../recovery_help` contains the shared prompt, capture, scoring and
allocation functions needed by these experiments.
