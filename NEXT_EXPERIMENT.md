# Next experiment: match a failure to a remedy

**Status: completed with revisions; this is the original proposal.**

The [implemented protocol](experiments/paired_evidence/README.md) and
[heldout results](results/paired_evidence/README.md) supersede this plan.
The proposed same-action distance has a deletion-collapse degeneracy: emptying
evidence makes both compared inputs identical even when both answers are wrong.
The learned condition and strategy-transition policies did not pass every
control on both backbones. The text below preserves the original hypothesis;
it is not a claim that its geometry criterion worked.

## Question

For the same underlying question, does a useful remedy selectively reverse the
activation change associated with an introduced problem? If so, can we estimate
that relationship from a single initial execution?

## Matched questions

Use an independent calibration set to choose a task where the model is reliably
correct when supplied with sufficient evidence. For each base question, create:

| Version | Change | Hypothesized useful action |
|---|---|---|
| Clean | Sufficient relevant evidence | Keep |
| Missing evidence | Remove a necessary fact, retaining it in the retrieval corpus | Retrieve more evidence |
| Distractor interference | Keep the necessary facts and introduce plausible irrelevant evidence | Filter evidence |

Keep the question and gold answer unchanged. Match length and format with
harmless replacement passages. Apply every action to every version, including
clean cases. Remedies must operate from public inputs, without seeing gold
answers, supporting-fact annotations or the corruption label. Do not choose
questions based on whether the desired remedy succeeds.

First establish distinct behavioral benefits. If one remedy dominates both
conditions, this action menu cannot establish useful strategy matching. Report
all cases as well as the baseline-wrong subgroup, and measure damage to initially
correct answers. The conditions identify input manipulations, not unique
internal failure mechanisms.

## Offline geometry check

Measure activations at a fixed answer-start position before generation, across
predeclared layers. Fit compression and normalization only on calibration data.
For a question, let `h_clean` and `h_bad` be its matched activations, and use
suffix `a` for the corresponding states after applying a remedy.

```text
failure displacement: d = h_bad - h_clean
remedy-specific change: m_a = (h_bad,a - h_bad) - (h_clean,a - h_clean)
gap reduction: S_a = ||d||² - ||d + m_a||²
```

Subtracting the strategy's effect on the clean run helps control common prompt
movement. It does not automatically isolate a causal mechanism. Check whether
larger gap reduction predicts actual answer repair, and whether the clean
counterpart remains correct after the action. Collapsing both states to the
same wrong answer must not count as repair.

Include mismatched remedies, shuffled directions, harmless prompt changes, and
confidence/text/risk-only controls. The exact clean state and all action states
are available only for this offline diagnostic. They cannot be inputs to a
claimed single-shot method.

## Single-shot check

Only if the offline relationship holds, fit small predictors of `d` and `m_a`
from the initial activation. Training targets are paired activation differences;
mitigation correctness must not select predictors, layers or hyperparameters.
Offline strategy executions are still needed to measure their changes.

At test time, save the chosen action before executing any candidate remedy.
The predictor receives no clean counterpart, corruption label, gold answer or
post-remedy activation. Compare against strong fixed and risk-gated policies at
matched cost, plus text/confidence baselines with clearly stated supervision.

## Scale and interpretation

Start with 200 distinct base questions on one model for feasibility. Use the
observed variance to set a fresh confirmatory sample size and a meaningful
effect threshold before inspecting test outcomes. Keep every variant of a base
question in the same split; separate topics/templates where feasible. Freeze
the design and replicate on two models. Intervals should resample base questions,
not treat their variants as independent observations.

- No remedy specificity: change the task/action menu before testing a selector.
- Remedy specificity but no geometry/repair relationship: evidence against this
  distance-to-clean rule under the tested setup.
- Offline geometry works but initial-state prediction fails: reference or action
  estimation is the bottleneck.
- Single-shot selection reliably beats strong controls: useful evidence for the
  method, followed by a separate test on naturally occurring errors.

A claim about causal internal mechanisms would additionally require controlled
activation injection/ablation. Novelty and broad generality are separate questions.
