# Pre-registration: LLM-judge audit of the answer check

Committed before any judge outputs existed.

## Why

Every result uses a containment check (an answer counts as correct if an accepted gold answer appears in it).
This can (a) mislabel answers, and (b) let correct paraphrases slip into the "sampled wrong" targets of the
type-matched control, which would inflate the headline ("plausible wrong answers are easier to reach than gold").

## Judges

Two local instruction models: **Qwen2.5-7B-Instruct** (primary) and **Phi-3.5-mini-instruct** (independent,
not one of the 10 evaluated models). Prompt: question, the reference answers, the proposed answer; "Is the
proposed answer correct, i.e. does it refer to the same thing as one of the reference answers? Reply with only
yes or no." Verdict = p(yes) > p(no) from next-token logits.

## Audit A: label agreement

Per dataset (TriviaQA, NQ-Open), 50 greedy answers per model (seed 3), 500 per dataset. Report the agreement and
Cohen's kappa between containment and each judge, and the false-positive and false-negative rates of containment
against the primary judge.

## Audit B: are the "sampled wrong" targets really wrong?

All sampled-wrong targets of the type-matched control (both datasets), plus 200 gold targets as a sanity check
(the judges should accept these). Report the share of sampled-wrong targets that a judge calls correct.

**Headline robustness (decision rule).** Recompute gold vs sampled-wrong at ε = 0.05 using only questions whose
sampled-wrong target **both judges** call wrong. The headline survives on a dataset if gold is **not**
significantly easier than sampled-wrong in ≥ 7/10 models (same rule as `prereg_nq_replication.md`).

## Deviations

(none yet)
