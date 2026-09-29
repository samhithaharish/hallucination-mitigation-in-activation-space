# FACR / LAPSH: Detectability != Correctability

Code for the ICML 2027 submission. Core claim: **the layer where a linear probe best detects a
hallucination is not the layer where activation steering best corrects it.** On Llama-3.2-3B the
detection peak is around layer 13 (~46% depth) and the steering peak around layer 24 (~86% depth);
the same gap was seen on Qwen-2.5-3B and Mistral-3B. The open question for reviewers is whether it holds
at 8B scale.

## Layout

```
src/facr/
  config.py      model registry + RunConfig (no secrets; HF token comes from the environment)
  data.py        TriviaQA generation + token-F1 labelling (label 1 = hallucinated)
  features.py    per-layer mean-pooled hidden states, disk-cached
  probes.py      per-layer logistic-regression AUROC sweep; boundary-case selection
  steering.py    CAA vectors, forward hook, steering evaluation, steer-layer sweep
  ablation.py    layer-swap grid (decouple detect layer from steer layer)
scripts/
  run_experiment.py   end-to-end run for one model -> results/cross_model_comparison.csv
  clean_notebook.py   strip outputs + scan notebooks for secrets before committing
docs/notebook_inventory.md   which Colab notebook became which module, Drive IDs, what is verified
tests/test_steering.py       CPU smoke tests on a tiny random Llama (hook placement, shapes, cleanup)
notebooks/archive/           original Colab notebooks (git-ignored until cleaned)
results/                     cross_model_comparison.csv; cache/ is git-ignored
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # then put a NEW Hugging Face token in it
export $(grep -v '^#' .env | xargs)

python scripts/run_experiment.py --model llama3.1-8b     # ~40GB GPU (one A100)
python scripts/run_experiment.py --model qwen2.5-7b
```

Every stage caches to `results/cache/<model>/`, so a crashed run resumes where it stopped.
Layer indices follow the `hidden_states` convention: 0 is the embedding output, k is the output of
block k-1. Steering at layer k hooks block k-1.

In Colab, do not `pip uninstall torchvision`; that is what broke the 8B run (torch/torchaudio CUDA
mismatch on model import). If you hit it again, reinstall torchaudio to match torch.

## Status vs. what reviewers will ask

| Item | Status |
|---|---|
| Detect/steer gap on Llama-3.2-3B, Qwen-2.5-3B, Mistral-3B | done per your notes; numbers need to go in `results/cross_model_comparison.csv` |
| Layer-swap ablation | notebook exists (`Layer_Swap_Ablation (1).ipynb`, Jul 4); results not yet checked here |
| **8B scale (Llama-3.1-8B / Qwen-2.5-7B)** | **not done.** The Jul 21 run crashed at model load; cells 4 to 12 never ran |
| Cross-dataset (second benchmark beyond TriviaQA) | not started as far as I can tell (`LAPSH_GSM8K` is a starting point) |
| Causal check (activation patching, Zhang & Nanda) | optional; discussed, not implemented |
| Why the gap exists (theory / intuition) | not started; reviewers will want one paragraph |

## Methodology issues to settle before the numbers go in the paper

1. **Steering metric.** The original notebooks counted a case as "correct" when the steered answer
   *differed* from the original generation, which rewards any change. `eval_steering_accuracy` now
   scores against gold aliases. Numbers from older notebooks are not comparable, so rerun the 3B models
   with this code.
2. **Fixed alpha, unnormalized vectors.** Raw mean-difference vectors have layer-dependent norms, so
   alpha = 4 is a different perturbation strength at each layer, which can itself create a "best steer
   layer". `--normalize-vectors` gives a comparable strength; report both, or sweep alpha per layer.
3. **Boundary cases.** They are chosen by probe uncertainty over all samples, including the probe's own
   training split. Choose them from the held-out split only.
4. **Sample size.** Scoring a steer layer on 20 to 30 examples gives very noisy accuracies (one example is
   3 to 5 points). The argmax over ~30 layers will pick noise. Use more boundary cases, several seeds and
   confidence intervals before claiming a peak.

## Security

Never commit tokens. `.env` is git-ignored. Run `python scripts/clean_notebook.py <nb> --check` on any
notebook before it goes into git; it exits non-zero on anything that looks like a credential.
