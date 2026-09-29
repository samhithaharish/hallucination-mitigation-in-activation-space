# Method design (draft, 2026-09-29): read-here, write-there knowledge-aware correction

Status: proposal. Built only if the correctability pilot supports it (fixable errors exist, and ideally
the cheapest-to-fix layer S* differs from the detection layer D).

## Idea in one sentence

Read the model's state at the layer where hallucination is most detectable (D), decide per question
whether to **keep**, **correct** or **abstain**, and when correcting, write a question-specific edit at
the layer where correction is cheapest (S*), trained to reproduce the model's **own** state from runs
where it answered correctly.

## Components

1. **Detector** at read layer D: p(hallucinate | h_D). Already built (D = 13 on Llama-3.2-3B).
2. **Knowledge head** at D: p(model knows | h_D). Trained on train questions labelled by sampling
   (knows = at least 2 of 10 samples at T = 1 correct). Separates fixable errors (knows) from knowledge gaps.
3. **Corrector** g(h_D) → δ, added at write layer S*: a small low-rank MLP (input-dependent, not one
   fixed vector, which H4 in the pilot should show cannot work).
   - Target (the novel part, "self-state distillation"): for questions the model knows, the state at S*
     from the model's own correct sampled run, plus the NLL of that self-generated correct answer.
     Gold labels are needed only to decide which training samples were correct; at test time nothing
     external is needed and no answer text is injected.
4. **Gate** π(keep / correct / abstain | h_D), trained on the reward +1 correct, 0 abstain, −λ wrong.
   All three outcomes can be observed for every training question, so a cost-sensitive classifier
   solves this one-step decision exactly. **Optional RL stage**: fine-tune the corrector and gate
   end-to-end with policy gradient on sampled decoded answers, so the objective is real decoded
   correctness rather than a teacher-forced proxy.

## What is new relative to prior work

| Prior work | What it does | Difference here |
|---|---|---|
| ITI, CAA | one fixed direction for all inputs | input-dependent edit |
| TruthFlow (2502.04556) | question-specific correction via flow matching, trained on TruthfulQA truthful/untruthful pairs | targets are the model's own correct states (not external answers); read and write layers decoupled; explicit knows / doesn't-know gate with abstention |
| ReFT, LoFiT | learned representation fine-tuning at chosen positions and layers | intervention placed by the measured detect/correct maps, gated by knowledge, and evaluated on honest open-QA scoring |
| CAST, SafeSwitch | probe-gated steering toward refusal | a three-way gate: keep / correct / abstain |
| Commit-Abstain Circuit (2609.32964) | abstention circuit and policy | adds correction when the model knows, not only abstention |
| Detection-without-correction papers (2604.13068, 2603.18353, 2605.05715) | fixed steering fails | explains why (pilot H3/H4) and gives a method that works within the fixable subset |

## Evaluation (pre-registered per model before running)

- Held-out questions only. Metrics: accuracy; harm rate (correct → wrong); abstention rate;
  **reliability = correct − λ·wrong** for λ ∈ {1, 2, 4}; risk-coverage curve.
- Upper bound: the per-question oracle fix rate from the pilot (no method can beat it).
- Baselines, all tuned on the same tune split:
  - no intervention
  - CAA; ITI
  - CAST-style gated steering
  - TruthFlow
  - ReFT / LoFiT
  - LoRA fine-tuning on the same training questions ("why not just fine-tune?")
  - **detector-only abstention** (the critical one: correction must add beyond abstaining)
  - semantic-entropy / self-consistency abstention
- Ablations:
  - read = write at D vs decoupled
  - gold teacher-forced target vs self-state target
  - no knowledge head
  - fixed vector vs input-dependent
  - with and without the RL stage
- Scale:
  - models: Llama-3.2-3B, Llama-3.1-8B, Qwen-2.5-3B/7B, Gemma-2-2B/9B
  - datasets: TriviaQA, NQ-Open, PopQA (long-tail, knowledge-gap heavy)
  - out-of-distribution transfer: train on TriviaQA, test on NQ-Open and PopQA

## Honest ceiling

A corrector can only fix errors the model "knows". If about 38% of answers are wrong and a quarter to a third
of those are "knows" errors, the accuracy gain is at most about 10 points and realistically 3–5. Most of
the reliability gain will come from abstaining on knowledge gaps. The paper stands only if the method beats
**detector-only abstention** and **LoRA fine-tuning** at equal coverage.

## Kill criteria

- Pilot: the per-question oracle fixes < 10% of errors at modest ε, or gold ≈ control → stop, no corrector.
- Stage 1 (Llama-3.2-3B): no gain over detector-only abstention in reliability at λ = 2 → stop.
