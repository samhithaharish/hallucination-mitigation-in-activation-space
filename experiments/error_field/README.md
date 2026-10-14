# Forecasting error-risk changes from initial activations

This separate experiment addresses a limitation of the paired evidence study:
recognizing an input perturbation is not the same objective as repairing an
answer. Its design starts from development data, before inspecting the preceding
study's heldout answer-quality results. It uses a new heldout question cohort.
It is exploratory method development, not a claim of a novel or confirmed ICML
result.

1. Fit an error probe on the original answers only. Inputs are the existing
   paired-difference activation features at the last prompt token. Targets are
   whether the ordinary baseline answer exactly matches the gold answer. Choose
   probe regularization by baseline validation log loss. This step uses baseline
   correctness supervision; it is not unsupervised.
2. Freeze that readout. On offline strategy activations, measure its logit change
   for keep/retrieve/filter/focus and sham/empty controls. Fit response models
   from the initial state to these changes. No strategy answer is scored during
   this fitting. Compare mean, linear and bilinear models using validation
   logit-transition error, with a shared configuration across backbones.
3. Select the largest predicted reduction in error probability, with a one-
   standard-deviation validation residual penalty on each action's predicted
   error logit. Keep has zero gain. This is a heuristic upper-risk penalty, not
   calibrated uncertainty. Sham/empty predictions remain diagnostic controls.
4. Give public-text/pool/confidence and confidence-only forecasters the same
   frozen logit-transition targets and validation selection. Also compare the
   earlier direct condition rules. These controls test whether initial
   activations contribute beyond free observable features.
5. Freeze models and new-test decisions before any new-test mitigation call.
   Evaluate F1, EM, rescue, harm, calls/tokens, risk-shuffled routing, strongest
   risk-gated fixed action/threshold within the same call budget, and the direct
   condition rules. All variants of a question stay together in resampling.
   Report the same method on both backbones. No test outcomes tune the model.

The key assumption is that a baseline error probe remains useful on modified
prompts. Predicting its transitions accurately does not establish that assumption;
heldout answer quality and deletion controls must test it. The study may fail.
Recovery always means recovery by the available action bank, not universal
recoverability. The fixed action bank and condition maps were designed with
independent pilot outcome information; learned transition/configuration fitting
receives no mitigation-success targets.

## Mathematical target and limitation

Let q(h) be the baseline-trained error logit. For action a, learn
delta_a(h) = q(h_after,a) - q(h) from activation pairs. At inference choose the
smallest predicted post-action logit q(h) + predicted_delta_a(h) + sigma_a;
keep has delta and sigma zero. The common q(h) cancels in ranking. Sigma is the
validation root mean squared transition error, not a confidence certificate.

There are two distinct requirements: accurately forecast the readout's change,
and have that readout track true error after each action. Under the **assumptions**
that sigmoid(q_after,a) approximates true conditional error risk within epsilon_p
for every action, and the penalized forecast approximates q_after,a within
epsilon_d, the chosen action's excess error risk is at most

    2 * epsilon_p + epsilon_d / 2.

This follows from the sigmoid's 1/4 Lipschitz bound and the standard two-error
argument for minimizing an approximate objective. The experiment does not prove
these uniform assumptions. In particular, small activation-transition MSE does
not establish epsilon_p; heldout probe transfer and answer outcomes are necessary.
No novelty claim rests on this elementary bound.

## Run

Set ACTIVATION_RUNS to a fresh run root. Run `python -m experiments.error_field.data
--source PREVIOUS_RUN_ROOT/paired_evidence`; the previous paired study must have
completed train/validation fitting so its coordinates and best model exist.
The source development executions are read through symlinks and are not modified.
Commit the method, then run `python -m experiments.error_field.fit`. It compares
the prespecified 13 configurations using activation-transition error only and
freezes a router for each model. Its configuration table is saved under
`ACTIVATION_RUNS/error_field/selection.json`.

Collect each backbone with `python -m experiments.paired_evidence.collect
--model MODEL --splits test --predictor
ACTIVATION_RUNS/paired_evidence/MODEL/fitted/best.joblib`, using the established
GPU/PYTHONHASHSEED environment. Then run `python -m experiments.error_field.evaluate
--model MODEL` for both, followed by `python -m experiments.error_field.report`.
Both model/source hashes and pre-mitigation prediction timestamps are checked.
Article-title disjointness is relative to the source v3 candidate pools; all
prior question IDs are excluded, while title overlap with the earlier v2
qualification cohort is not guaranteed.
