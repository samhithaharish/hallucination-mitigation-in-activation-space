# Activation drift for choosing a recovery strategy

We are studying whether a language model's initial activations can tell us
**which recovery strategy is likely to help**: retrieve more evidence, reason
again, filter evidence, or keep the current answer.

The current direction is to learn how strategies change activations, then match
those changes to the model's activation drift. We want to do this without
training the strategy chooser on labels saying which mitigation succeeded.
Offline calibration runs are still needed to measure strategy effects.

**Current status: the experiments are complete, but reliable selection from
activation drift has not been demonstrated.** Repeatable average strategy
changes are measurable; they are not yet established repair directions.

## Start here

- [Results and what they mean](RESULTS.md)
- [Current method and code](experiments/activation_drift/README.md)
- [Proposed next experiment](NEXT_EXPERIMENT.md) — not yet implemented or run
- [How to reproduce](REPRODUCE.md)

## Completed comparisons

| Experiment | Question | Finding |
|---|---|---|
| Supervised recovery selection | Do initial activations improve prediction of labelled strategy gains? | Improved over behavioral features on Qwen; no replicated gain on Phi |
| Activation-drift selection | Can predicted strategy movements identify a useful remedy without strategy-success labels? | Improved over keeping the initial answer, but did not establish an advantage over strong fixed-strategy controls |

The models were Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct on HotpotQA
*distractor*. Recovery selection used 2,000 matched test questions per model;
the later drift study used 1,000 fresh test questions per model. Details,
confidence intervals and limits are in [RESULTS.md](RESULTS.md).

## What is in this branch

```text
experiments/
  activation_drift/     Current direction-based experiment and tests
  recovery_scale/       Earlier supervised comparison and tests
  recovery_help/        Shared prompts, capture, scoring and budget allocation
  paths.py              Configurable paths and pinned model/data versions
  prepare.py            Restore the exact recorded cohorts and freeze run sources
results/
  activation_drift/     Recorded summary, figure and cohort IDs
  recovery_scale/       Recorded summary, figure, cohort IDs and model checksums
  recovery_pilot/       Prior cohort exclusions and independent scorer checksum
```

Raw activations, model weights, fitted predictors and generated answers are kept
outside Git. The small committed result files let readers inspect the findings;
full reproduction needs the dependencies, data and GPU runs described in
[REPRODUCE.md](REPRODUCE.md).

## Next scientific question

For matched clean, missing-evidence and distractor versions of the same question,
does a remedy reverse the failure-associated activation change, and does that
reversal track actual answer repair? Establish that relationship before learning
a predictor that must work from a single initial execution. The
[next-experiment document](NEXT_EXPERIMENT.md) specifies the controls and the
conditions under which we should stop or proceed.

## Earlier detection and steering work

The repository's existing `scripts/`, `src/`, notebooks and older results contain
the earlier detection/steering study that motivated this project. Its
[previous overview](https://github.com/samhithaharish/hallucination-mitigation-in-activation-space/blob/04f07bd349e0fd188525bc068d1b3976fcced3d5/README.md)
remains available in history. The current experiment does not establish a
causal faulty layer, a universal repair direction, or universal irrecoverability.
