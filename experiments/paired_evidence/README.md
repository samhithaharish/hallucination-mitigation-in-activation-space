# Choosing an evidence repair from initial activations

This experiment asks whether controlled evidence perturbations can teach a small
activation model to choose between adding evidence and filtering evidence on
new questions. It is a narrower task than identifying every cause of an LLM error.
The current experiment is in progress; no positive result is asserted here.

Each HotpotQA base question has three paired versions: two supporting articles
plus unrelated filler; one supporting article replaced by filler; supporting
articles plus the original related distractors. Questions and gold answers stay
fixed. Supporting annotations construct the benchmark only. Approximate word
lengths and passage counts are matched; passage order is shared within a base.
Every variant stays in its base's split. Invalid supporting-sentence annotations
are excluded before construction. Hard questions are excluded for this capability-
qualified pilot; this restricts generality. Earlier experiment questions are excluded.

The public input contains only a question, current evidence and a candidate
article pool. Recovery actions receive no labels or support annotations. Retrieval
uses an LLM-written query to add up to four BM25-ranked candidate articles.
Filtering uses an LLM to select up to four current articles; invalid selections
fall back to a declared BM25 rule. Each active strategy uses two generations.
An evidence-preserving focus action moves the selected articles to the end while
retaining every passage. Keep, a harmless instruction change, and empty evidence
provide controls. Keep/retrieve/filter/focus can be selected by the policy. The
filter/focus branches share their identical public passage-selection preprocessing
during offline collection; either would cost two calls when selected alone.

States are measured at the last prompt token at six fixed fractional depths,
before any answer tokens are supplied to the measurement model. Generations use
the same pinned vLLM backbones as the earlier study; exact prompt token IDs are
replayed with Transformers. The scientific coordinate differs from the previous
near-end-of-answer measurement. Output quality and parse failures are recorded.

The offline diagnostic checks behavior and activation-gap contraction, including
whether an action collapses two states while making both answers wrong. A clean
counterpart is experimental information and cannot be supplied to a deployed
selector.

The proposed predictor learns two evidence-condition coordinates from paired
activations (clean=(0,0), missing=(1,0), distractor=(0,1)). These targets describe
input interventions, not established internal mechanisms or correctness. It then
learns how strategies move those coordinates, using only activation pairs.
Regularization and action models are selected by condition/displacement error,
without mitigation-success labels. Final choices are saved before test actions
execute. One initial execution supplies the selection inputs.

Independent calibration and a fresh feasibility cohort precede predictor fitting.
Fresh fitting/validation/test cohorts contain 400/100/800 base questions. A
separate natural-question cohort is reserved for a later transfer check. All
answer-quality comparisons resample base questions to keep paired variants
together. Report fixed and risk-matched controls, observable baselines, rescue,
harm, costs, and oracle action headroom. Controlled success would require separate
natural-error and cross-task verification. No observed repair does not establish
universal irrecoverability.

Implementation: task.py defines construction and public actions; data.py freezes
cohorts; collect.py captures states and unscored outputs; fit.py fits activation
models; evaluate.py scores outputs after method freezing. The local autoresearch
loop is bounded to 12 activation-model comparisons, and cannot optimize final
mitigation success. Method updates after a failed test require a new cohort and
an explicit new experiment record.

Independent 32-base calibration exposed harmful evidence removal by filtering.
Focus was added before observing the fresh 200-base feasibility cohort. Original
calibration outputs remain under the v1 run root; subsequent runs use v2.
