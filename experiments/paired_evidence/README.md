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
calibration outputs remain under the v1 run root; v2 was exploratory qualification; the corrected frozen experiment uses v3.

## Frozen v3 protocol

The first 200-question qualification cohort exposed a public-input shortcut:
current-passage membership in the old candidate pool revealed the condition.
Those questions and all other v2 cohorts are excluded from v3. They provide
exploratory checks only. V3 uses the same shuffled 20-article union pool across
all three variants; all ten current articles appear in that pool. Background
filler titles are disjoint across splits and absent from every selected source
question's context. Candidate-pool membership, BM25 coverage and scores are also
available to the observable comparator. Benchmark-construction choices can still
leave linguistic artifacts, so success is limited to these interventions.

For base question i, let h(i,c,a) contain the six prompt states for input condition
c and action a. Paired baseline differences h(i,c,keep)-h(i,clean,keep), including
the zero clean difference, define up to 16 principal directions per layer. A
linear map from projected initial states predicts the two condition coordinates
z. Its ridge strength is selected using validation condition MSE. Neither the
projection nor predictor receives answer-quality targets. At inference, the
clean counterpart and the condition identity are unavailable.

The action model predicts delta_a = z(h_after,a)-z(h_initial). Candidate models
are the unconditional mean and ridge predictors from z alone, the projected
initial states plus z, or those states with bilinear state-by-condition
interactions. The last feature map permits an action's effect to depend jointly
on context and evidence condition; it does not identify a causal layer pathway.
The three maps use ridge strengths 1, 10, 100 and 1,000. Twelve configurations are compared using
validation displacement MSE normalized by training displacement variance, with
a variance floor of 0.01. One configuration is chosen by mean NMSE across both
backbones. Each backbone has its own fitted parameters.

The learned strategy policy chooses the largest predicted reduction

    score(a) = ||z||² - ||z + predicted_delta_a||² - validation_error_a.

Validation error is the mean squared Euclidean displacement error; keep has
score zero. This is a conservative heuristic, not a probability of successful
repair. The frozen model also predicts sham/empty control moves with the same
configuration and saves its choices if those controls were allowed. These moves
are excluded from the search metric and deployed policy, and reported separately
to test whether the learned score also rewards information deletion. A separate condition policy uses the nearest condition coordinate and
the predefined map clean→keep, missing→retrieve, distractor→keep. The independent v3
pilot supported retrieval for missing evidence but did not establish a focus
benefit, so this sparse mapping was fixed before fitting or observing final test
outcomes. Observable comparators use the identical map. This simpler policy
tests whether detecting missing evidence is already sufficient.

Observable baselines predict the same condition targets from (1) prompt TF-IDF,
confidence, length and public pool metadata; or (2) confidence and length. They
receive the same validation ridge-strength selection. No policy is fitted using
which mitigation answered correctly. Baseline confidence requires the original
answer generation; selection never executes trial mitigations on the new input.
Predictions, IDs, model hashes and timestamps are saved before the first actual
mitigation preprocessing call, and checked again during scoring.

### Acceptance and interpretation

Two activation policies are prespecified: the learned strategy policy and the
condition policy. For a routing claim, the *same* policy must have a strictly
positive lower 95% paired-bootstrap bound against each of: keep, the strongest
fixed remedy within the same extra-generation-call budget, risk-bin-shuffled
routing, the public-input comparator, and the confidence comparator, on **both**
backbones. All variants are resampled together by base question. There are two
prespecified policy claims; using lower ends of two-sided 95% intervals gives a
nominal one-sided 2.5% threshold per policy (Bonferroni total 5%). These are
bootstrap approximations, not exact finite-sample error guarantees. Requiring
all comparators and both backbones is an intersection criterion. A claim that
learning strategy changes adds value additionally requires beating the simpler
condition policy. This distinction prevents calling condition detection evidence
for a useful learned strategy geometry.

The fixed-remedy control is chosen in hindsight and recomputed in each base
bootstrap. Its activation-risk gate receives the evaluated policy's extra-generation-call budget. It can use
retrieve/filter/focus (two calls each) or sham/empty (one call each); cheaper
actions may cover more inputs, up to all of them. Unused calls are allowed once
all inputs are covered. This matches a call-budget cap, not token count or latency. Also report the strongest always-on remedy, costs, rescue, harm, and
an oracle ceiling. A positive result would establish controlled evidence-repair
routing, not a universal hallucination detector, a causal layer pathway, natural
failure diagnosis, or proof that an unrepaired answer is irrecoverable.

## Run

Use the pinned local models/data and dependencies in the repository's main
reproduction instructions. Set ACTIVATION_RUNS to an empty run root. Execute
`python -m experiments.paired_evidence.data`, then collect each backbone with
`python -m experiments.paired_evidence.collect --model MODEL --splits pilot train validation`.
Use separate CUDA_VISIBLE_DEVICES for the two collectors. Review pilot feasibility
before freezing the predictor; do not fit or tune to pilot mitigation successes.

From a clean git worktree, with PYTHONHASHSEED=0 and the same ACTIVATION_RUNS,
run `python -m experiments.paired_evidence.search`. The bounded autoresearch loop
commits each predictor configuration, verifies activation NMSE, runs the task's
CPU tests, and keeps or reverts the change. Its ledger stays in the ignored
`autoresearch/loop-261013-paired` directory. It freezes fitted/best.joblib and
metadata separately for each model. Then collect test using `--splits test
--predictor RUN_ROOT/paired_evidence/MODEL/fitted/best.joblib`. Only afterward
run `python -m experiments.paired_evidence.evaluate --model MODEL --split test`.
Generation requires PYTHONHASHSEED=0; the established local commands also use
OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=4. Generated states and raw outputs
remain in the ignored run root; publish compact results and provenance.

## Closely related work

Activation-based retrieval decisions are established: [Probing-RAG](https://arxiv.org/abs/2410.13339)
uses intermediate hidden states to decide whether to retrieve again.
[Knowing Before Answering](https://arxiv.org/abs/2608.27661) trains linear models
on internal signals to distinguish sufficient, insufficient and conflicting
evidence across 16 backbones. [ProbeRAG](https://arxiv.org/abs/2510.12460)
combines context pruning, latent conflict detection and attention intervention.
Therefore, detecting an evidence condition from activations is not itself a
novel contribution. The unproven question here is whether paired strategy-state
changes, without mitigation-success labels, add useful repair selection beyond
a condition rule and strong observable controls. Empirical success and novelty
are separate requirements; this pilot cannot establish an ICML contribution.
