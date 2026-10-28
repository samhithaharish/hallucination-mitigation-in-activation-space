# Pre-registration: do activations encode the source of an error, beyond lexical and semantic confounds?

Written 2026-10-28, before any of the labels or no-context data below exist. Supersedes
`docs/prereg_error_types.md`: its labels were spot-checked and the "hallucination" class turned out to be mostly
over-general or memory-based answers (e.g. "India" for Kolkata), not inventions, so no probe was fitted on them.

## Question

The earlier probe asks *can I predict correct vs incorrect?* This study asks: **after controlling for lexical,
semantic and question-level information, do pre-generation activations encode not only whether an error will
occur, but its source?**

## Labels (same underlying questions; HotpotQA, Som's frozen cohorts)

| Label | Definition | How |
|---|---|---|
| **C** correct | means the same as the gold answer | gold contained in answer (rule), else judge "correct?" |
| **G** context-grounded error | wrong answer that the supplied passages explicitly support or assert as the answer | judge "do the passages support or assert the proposed answer as the answer to this question?" |
| **U** ungrounded error | wrong answer not supported by the supplied passages | the same judge question, answered no |
| **P** partially correct | captures some correct information but incomplete, too broad or underspecified (e.g. "India" for Kolkata) | judge "broader, narrower or incomplete version of the reference?" (asked before the G/U question) |
| excluded | empty answer, refusal, wrong yes/no answer | rules and judge, as in `error_types.py` |

- Judge: Qwen2.5-7B-Instruct, yes/no next-token logits, with the question, the two passages the model saw, the
  gold answer and the model's answer. A rule-based "answer string appears in the passages" flag is recorded and its
  agreement with G/U reported. 25 examples per label per model are written for a human spot check, and the labels
  are frozen after that check, before any probe is fitted.
- **P is kept out of all primary probes.** Afterwards, P examples are projected onto the learned error and
  grounding directions (s_error(P), s_ground(P)) and described.

## Data and features

- Models: Llama-3.2-3B, Phi-3.5-mini, Qwen2.5-3B, Llama-3.1-8B; `train` 6,000, `validation` 1,000, `test` 2,000.
- Activations h at every layer, **prompt end (pre-generation)**; answer end reported as secondary.
- **No-context ablation data:** the same questions, the same prompt with zero passages; answers and prompt-end
  states at every layer (`collect.py --keep-only --no-context`).

## Confound baselines (pre-generation inputs only: question + the passages shown)

1. TF-IDF (word 1-2 grams) of question + passages; 2. semantic: MiniLM sentence embedding of the same text;
3. question-level features: question length, type, level, passage lengths, BM25 score of the top passages.
Each is a standardised logistic regression with L2 strength chosen on `validation`.

## Hypotheses (test split; paired bootstrap 95% CIs over questions; 20-seed shuffled-label nulls)

- **H1 (primary):** for **G vs U** (errors only), the activation probe at the validation-chosen layer beats every
  confound baseline (AUROC difference CI > 0), **and** adding activations to all three baselines combined improves
  AUROC over the baselines combined (incremental-information test, CI > 0). Required in at least 3 of 4 models.
- **H2:** the activation G-vs-U AUROC (validation, per layer) increases with depth: Spearman correlation of AUROC
  with layer index > 0 over the first two thirds of the layers, and the best layer is not layer 1, in >= 3 of 4.
- **H3:** the grounding direction (G vs U) is distinct from the general error direction (C vs errors): at the
  shared best layer, |cosine| between the two standardised weight vectors is below the 5th percentile of their
  split-half self-cosines (each probe refitted on two disjoint halves of `train`), in >= 3 of 4 models.
- **H4 (generalisation; stage 2):** the G-vs-U probe transfers across models' own data (fit per model, layer by
  relative depth) and to a second dataset (2WikiMultihopQA) without refitting. Registered now; run after H1-H3.
- Reported alongside: C vs errors and the three-way C / G / U probe (macro AUROC), each with the same baselines.

## No-context ablation (descriptive, registered direction)

The G-vs-U probe is refitted on the **no-context** prompt-end states of the same questions. If G-vs-U separability
comes from how the model processes the passages, it should be clearly weaker without them (with-context minus
no-context AUROC, CI > 0). If it is just as strong, the signal is a question-level confound and H1 is reported with
that caveat.

## Class sizes

A model with fewer than 150 `train` examples of G or U is exploratory for that comparison (reported, not counted).
Failed hypotheses are reported as results. Deviations are logged below with dates.

## Deviations
- 2026-10-28 (pilot of the G/U judge question on 120 Llama-3.2-3B wrong answers, before any full labelling or
  probe): the judge called 117/120 wrong answers "not supported", including 85 whose answer string is in the
  passages. On clean HotpotQA passages the context almost never *asserts* a wrong answer, so natural G collapses.
  **Design change (agreed with the user):**
  - **Primary = controlled counterfactual contexts (B).** For each question answered correctly with its two gold
    supporting passages (O), build a matched counterfactual (CF): every occurrence of the gold answer in the
    passages replaced by a type-matched, plausible false answer (years/numbers by rule; entities proposed by
    Qwen2.5-7B and checked by rule: differs from gold, absent from the original passages, short); question,
    passage order and prompt otherwise identical. Labels under CF by rule: **G** (context-following) = output
    contains the planted answer and not the gold; **C** (context-resisting) = contains gold and not planted;
    **U** (other error) = neither. Primary probe: **G vs C from pre-generation (prompt-end) CF states**; then
    **Delta h_l = h_l(CF) - h_l(O)** as features for G vs C, compared with h_l(CF) alone and with the lexical,
    semantic and question-level baselines (H1-H3 restated on G vs C).
  - **Pilot gate:** ~200 O-correct validation questions on Llama-3.2-3B. Proceed to the full study only if both
    G and C have at least 40 examples in the pilot.
  - **Natural data (A), descriptive:** wrong answers under the original top-2 passages split by rule into
    context-copied (answer string in passages) vs non-context errors, with the same probes and baselines.
  - A neutral condition (answer-bearing sentence removed) is deferred until the O/CF pilot works.
