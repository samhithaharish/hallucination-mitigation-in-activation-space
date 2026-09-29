"""Model registry and run configuration.

Add a model here and every script picks it up via --model <key>.
Secrets (HF token) are read from the environment, never from source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    n_layers: int
    hidden_size: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # 3B family (already run; detect ~L13 / steer ~L24 on Llama-3.2-3B)
    "llama3.2-3b": ModelConfig("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", 28, 3072),
    "qwen2.5-3b": ModelConfig("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", 36, 2048),
    "mistral-3b": ModelConfig("mistral-3b", "TODO-fill-in-exact-hf-id", 0, 0),
    # 8B-class scale experiment
    "llama3.1-8b": ModelConfig("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", 32, 4096),
    "qwen2.5-7b": ModelConfig("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", 28, 3584),
}


@dataclass
class RunConfig:
    model_key: str = "llama3.1-8b"
    n_samples: int = 400          # TriviaQA questions used for factuality labelling
    n_boundary: int = 60          # rank-based boundary set size
    n_steer_eval: int = 30        # boundary cases used to score one steer layer
    n_swap_eval: int = 20         # boundary cases per cell of the layer-swap grid
    alpha: float = 4.0            # CAA steering strength
    f1_threshold: float = 0.3     # token-F1 below this => hallucinated
    max_new_tokens: int = 32
    test_size: float = 0.3
    seed: int = 42
    cache_root: str = "results/cache"

    @property
    def model(self) -> ModelConfig:
        return MODEL_CONFIGS[self.model_key]

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.cache_root, self.model_key)


def get_hf_token() -> str | None:
    """Read the token from the environment. Never hard-code it."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
