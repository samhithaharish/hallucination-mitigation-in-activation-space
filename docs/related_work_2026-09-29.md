# Related-work check (2026-09-29)

Search for prior work on each possible framing of this paper. The closest prior papers for each are listed.

| Framing | Status | Closest prior work |
|---|---|---|
| C. Detectable but not correctable | **Taken** | Roy et al., *Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing*, arXiv 2604.13068 (Mar 2026): probe-direction steering fails to correct in 7/7 models (GPT-2, Pythia, Qwen-2.5 up to 7B). Basu et al., *Interpretability without actionability*, arXiv 2603.18353 (Mar 2026): 98% probe AUROC, but steering (4 methods) leaves 76%+ of errors uncorrected. Liu, *Decodable but Not Corrected by Fixed Residual-Stream Linear Steering*, arXiv 2605.05715 (May 2026): 5 steering families, 29 configs, all Δ≈0; the probe used for abstention instead. |
| B. Detect-then-steer (probe-gated steering) | **Taken** | CAST, *Programming Refusal with Conditional Activation Steering*, arXiv 2409.05907 (ICLR 2025); SafeSwitch; AdaSteer 2504.09466. |
| Abstention steering, and the layer at which the commit/abstain decision is made | **Taken** | Nguyen et al., *The Commit-Abstain Circuit*, arXiv 2609.32964 (NeurIPS 2026): 10 models, 5 families; commit signals build early, abstention components act late; +12.2 pts decision accuracy. |
| Steering fixes only errors where the model knows the answer (HK+), not those where it doesn't (HK−) | **Taken** | Simhi et al., *Distinguishing Ignorance from Error*, arXiv 2410.22071; Simhi et al., *HACK*, arXiv 2510.24222 (uses steering to validate the knowledge axis). |
| Where detection peaks, and why simple probes work | **Taken** | Lee et al., *The Hallucination Signal Is a Mean Shift*, arXiv 2608.28930 (EMNLP 2026): signal spans a band of layers, LayerMix. |
| Evaluation metrics inflate reported gains | Partly taken (detection only) | *Re-evaluating Hallucination Detection in LLMs*, EMNLP 2025 (ROUGE vs LLM-judge). |
| **A. The best correction layer differs from the best detection layer (same model, same task)** | **Not found** | Nearest: *Dissociating Decodability and Causal Use* 2604.22128 (toy bracket transformers); the CAC timing result. Only possible if some steering method actually corrects (v2 run). |

Implication: A is the only framing still open, and it needs a steering method that works (v2).
