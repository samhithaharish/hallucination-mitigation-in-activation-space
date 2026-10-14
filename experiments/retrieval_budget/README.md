# Does the missing-evidence detector help at equal call budgets?

This is a **secondary exploratory analysis**, prompted by the completed first
paired-evidence test. It does not revise that experiment's failed primary
acceptance decision. Its detector and retrieval action are already frozen.
The first test suggested that the activation condition rule used fewer calls
than the public-input rule. Equal-budget public ranking was then examined
exploratorily. A fresh-cohort comparison is recorded before either ongoing
fresh test collection completes and before its answer outcomes are inspected.
No fresh outcome selects a policy, score, margin, comparator or threshold.

## Fixed comparison

Use the existing `class_rule`: classify the initial state as clean, missing or
distractor; retrieve only for missing. Its fitting uses intervention labels,
not mitigation-success targets. Pilot outcomes informed the action bank and
missing-only map. At inference it receives neither a condition label nor the
matched clean state. No weights are refit for this secondary analysis.

For the public and confidence condition probes, let d_c be squared distance to
condition coordinate c. Rank inputs by `min(d_clean,d_distractor)-d_missing`.
This favors missing evidence over either alternative. Rank activation drift by
its original `||z||²`, independently of the later baseline error probe.

Each control can choose **any one action** from retrieve/filter/focus/sham/empty
and its best affordable prefix in hindsight along its specified ranking. It
can leave calls unused. Retrieve/filter/focus cost two extra generation calls;
sham/empty cost one. The cap equals the evaluated detector's actual call count,
recomputed after every base-question bootstrap resample. Allowing action and
threshold choice from test outcomes makes this a strong evaluation benchmark;
it does not train the deployed policy. Also compare keep and shuffled routing
within ten activation-risk bins. All three variants of each question remain
in the same bootstrap draw.

For this single secondary policy, require a positive lower two-sided 95%
base-bootstrap bound against **all five** controls on **both** backbones.
These are nominal intervals. This analysis was motivated by preceding test
results and is one of several project hypotheses; it is not covered by the
original two-policy multiplicity statement and is not a confirmatory paper
claim. The fresh cohort supplies a separate replication check of this narrower
hypothesis. Report success or failure as recorded, including all controls.

## Registration and execution

From committed sources, with ACTIVATION_RUNS pointing to the new run root,
choose a fresh output path to preserve the published registration:

```bash
export RETRIEVAL_BUDGET_RESULTS="$ACTIVATION_RUNS/retrieval_budget_results"
python -m experiments.retrieval_budget.evaluate --freeze
```

This refuses registration after either new test collector completes. It saves
UTC time, code commit, evaluator/imported helper source hashes, cohort hash,
frozen predictor hashes and the fixed 4,000-draw/seed/bin settings in
`RETRIEVAL_BUDGET_RESULTS/registration.json` (the default is
`results/retrieval_budget/registration.json`). Keep the override set for every
secondary evaluation/report command. This output-path option was added after the
recorded run; original hashes remain unchanged. For replay of its local artifacts,
use the registered checkpoint described in the root reproduction guide. The primary error-logit and
clean-reference studies, their source hashes, and their saved policies remain
unchanged.

After collection, score both the original exploratory run and the new run
before building the combined report. Keep ACTIVATION_RUNS set to the new root
except for the two explicit overrides (replace the example original path):

```bash
ACTIVATION_RUNS=/path/to/original/run-root python -m experiments.retrieval_budget.evaluate --model qwen7b --label exploration
ACTIVATION_RUNS=/path/to/original/run-root python -m experiments.retrieval_budget.evaluate --model phi35 --label exploration
python -m experiments.retrieval_budget.evaluate --model qwen7b --label replication
python -m experiments.retrieval_budget.evaluate --model phi35 --label replication
python -m experiments.retrieval_budget.report
```

Set ACTIVATION_RUNS to the first paired run root and use `--label exploration`
to save the original exploratory comparison. The registered source is required
for both labels; replication additionally checks its exact cohort/model hashes.
Publication exports remain in the active checkout's results directory.

Extra-call budgets do not measure tokens, latency, peak memory or the cost of
capturing/replaying activations. No claim of deployment speed follows from
this comparison. Activation-based retrieval and evidence-condition probes
already have prior work; novelty and natural-error generality are separate.
