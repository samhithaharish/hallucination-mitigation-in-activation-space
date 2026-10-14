# Activation drift for choosing a recovery strategy

We are studying whether a language model's initial activations can tell us
**which recovery strategy is likely to help**: retrieve more evidence, reason
again, filter evidence, or keep the current answer.

The current direction is to learn how strategies change activations, then match
those changes to the model's activation drift. We want to do this without
training the strategy chooser on labels saying which mitigation succeeded.
Offline calibration runs are still needed to measure strategy effects.

**Current status: a fresh paired-evidence experiment is complete.** Detecting
missing evidence and retrieving selectively improved F1 on both backbones, but
neither activation policy passed all controls on both. Accurately predicting
strategy changes did not establish better repair selection.
See the [heldout results](results/paired_evidence/README.md).

## Start here

- [Results and what they mean](RESULTS.md)
- [Current paired-evidence method](experiments/paired_evidence/README.md)
- [Paired-evidence results](results/paired_evidence/README.md)
- [Earlier drift method](experiments/activation_drift/README.md)
- [Original paired-experiment proposal and outcome](NEXT_EXPERIMENT.md)
- [How to reproduce](REPRODUCE.md)

## Completed comparisons

| Experiment | Question | Finding |
|---|---|---|
| Supervised recovery selection | Do initial activations improve prediction of labelled strategy gains? | Improved over behavioral features on Qwen; no replicated gain on Phi |
| Paired evidence interventions | Can paired states teach repair selection without mitigation-success targets? | Neither strategy transitions nor the condition rule beat every control on both backbones |
| Activation-drift selection | Can predicted strategy movements identify a useful remedy without strategy-success labels? | Improved over keeping the initial answer, but did not establish an advantage over strong fixed-strategy controls |

The models were Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct on HotpotQA
*distractor*. Recovery selection used 2,000 matched test questions per model;
the drift study used 1,000 fresh test questions per model. The newest study used
800 fresh base questions with three paired evidence variants each, per model. Details,
confidence intervals and limits are in [RESULTS.md](RESULTS.md).

## What is in this branch

```text
experiments/
  paired_evidence/      Paired conditions, activation-only model search and tests
  activation_drift/     Earlier direction-based experiment and tests
  recovery_scale/       Earlier supervised comparison and tests
  recovery_help/        Shared prompts, capture, scoring and budget allocation
  paths.py              Configurable paths and pinned model/data versions
  prepare.py            Restore the exact recorded cohorts and freeze run sources
results/
  paired_evidence/      Heldout comparisons, search ledger, hashes and cohort IDs
  activation_drift/     Recorded summary, figure and cohort IDs
  recovery_scale/       Recorded summary, figure, cohort IDs and model checksums
  recovery_pilot/       Prior cohort exclusions and independent scorer checksum
```

Raw activations, model weights, fitted predictors and generated answers are kept
outside Git. The small committed result files let readers inspect the findings;
full reproduction needs the dependencies, data and GPU runs described in
[REPRODUCE.md](REPRODUCE.md).

## What the new experiment changes

The original proposal minimized the distance between equally modified clean and
corrupted states. Emptying their evidence makes both inputs identical and
maximizes that score, even when both answers become wrong. This is a concrete
failure of that distance criterion. The tested learned condition/transition
policies avoid selecting empty evidence, but did not establish a replicated
advantage over all controls. A useful repair objective must track answer quality
beyond merely closing an activation gap.

## Earlier detection and steering work

The repository's existing `scripts/`, `src/`, notebooks and older results contain
the earlier detection/steering study that motivated this project. Its
[previous overview](https://github.com/samhithaharish/hallucination-mitigation-in-activation-space/blob/04f07bd349e0fd188525bc068d1b3976fcced3d5/README.md)
remains available in history. The current experiment does not establish a
causal faulty layer, a universal repair direction, or universal irrecoverability.
