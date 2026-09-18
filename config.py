from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    adapter_rank: int = 8
    adapter_alpha: float = 8.0


@dataclass
class TrainConfig:
    dataset: str = "wikitext2"
    data_dir: str = "data"
    output_dir: str = "runs"
    run_name: str = ""
    embedding_type: str = "tied"
    seed: int = 1337
    device: str = "auto"
    steps: int = 2000
    batch_size: int = 2
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    save_interval: int = 500


def as_dict(model_config: ModelConfig, train_config: TrainConfig) -> dict[str, Any]:
    return {"model": asdict(model_config), "train": asdict(train_config)}
