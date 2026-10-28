# Pre-registration: do hallucination and concept drift have distinct activation patterns?

Written 2026-10-28, before any of the labels below exist.

## Question

The existing probe separates "answer matches gold" from "doesn't" (F1 >= 0.5), which mixes several error
types. We ask whether two specific error types have **distinct** internal patterns:

| Type | Definition | Example (Q: capital of Australia?) |
|---|---|---|
| **correct** | means the same as the gold answer | "Canberra" |
| **concept drift** | wrong, but a real, closely related concept: another entity from the question or passages, or a related entity in the wrong role | "Sydney" |
| **hallucination** | wrong and unsupported: invented or unrelated, not in the passages and not a closely related real concept | "Port Melville" |
| **refusal** | says it cannot answer or does not know | "The passages don't say" |

## Data

- Models: Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-3B, Llama-3.1-8B. HotpotQA, Som's frozen cohorts.
- The baseline (keep) answers already collected: `train` 6,000, `validation` 1,000, `test` 2,000 per model
  (Llama-3.1-8B `test` collected with `collect.py --keep-only` before labelling), with prompt-end and
  answer-end states at every layer.

## Labelling

- Judge: Qwen2.5-7B-Instruct, one 4-way multiple-choice prompt (question, the two passages the model saw, gold
  answer, model answer, the four definitions above); the label is the option with the highest next-token logit.
- Checks before any probe is fitted: (a) agreement with Laya on a 1,000-answer sample (4-way `choice`);
  (b) a random 25 per label per model written to `results/error_types/spot_check_<model>.md` for a human
  read; (c) agreement of "correct" with the existing binary judge. The labels are frozen after these checks.
- Qwen2.5-3B is judged by a same-family model; its results are flagged and repeated with Laya labels.

## Tests (per model; test split; paired bootstrap 95% CI)

Features: activations at each layer, prompt end (before the answer) and answer end (after it). Probes:
standardised logistic regression, L2 strength and layer chosen on `validation`.

- **E1 (distinct, primary):** a probe separating **hallucination vs concept drift** (wrong answers only) has a
  test AUROC with CI above 0.5 **and** above every surface baseline below, in at least 3 of 4 models.
- **E2 (each is detectable):** hallucination-vs-correct and drift-vs-correct probes, AUROC CI > 0.5.
- **E3 (different directions):** cosine between the E2 probe weight vectors at a shared layer, compared with a
  split-half cosine of each probe with itself (two disjoint halves of `train`).
- Surface baselines, fitted the same way on the same split: (i) TF-IDF of question + passages (+ answer for the
  answer-end comparison); (ii) simple features: question length, type, level, answer length, whether the answer
  appears in the passages; (iii) MiniLM sentence embedding of the same text; (iv) the model's 13 confidence
  features. Plus "surface + activations" vs "surface alone" (does adding activations help).
- Shuffled-label null (20 seeds) for each primary probe.
- Descriptive: label counts per model; AUROC by layer for E1 and E2.

A class with fewer than 150 `train` examples makes that model's test exploratory (reported, not counted).
Failed hypotheses are reported as results. Deviations are logged below with dates.

## Deviations
