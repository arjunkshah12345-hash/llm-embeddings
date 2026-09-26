"""Kaggle-only data and architecture probe for frozen Study 3 settings."""

from __future__ import annotations

import json
from pathlib import Path

from config import ModelConfig
from data import TokenDataset
from model import GPTModel


def main() -> None:
    root = Path("/kaggle/working/scale_probe")
    data = TokenDataset("/kaggle/working/scale_data", "fineweb_edu", seed=1337)
    data.save_metadata()
    model_config = ModelConfig(
        block_size=512,
        n_layer=12,
        n_head=12,
        n_embd=768,
        adapter_rank=8,
        adapter_alpha=8,
        capacity_control_width=532,
    )
    counts = {}
    for condition in ("tied", "partial", "untied", "capacity_control"):
        model = GPTModel(model_config, condition, seed=1337)
        counts[condition] = model.parameter_counts()
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.json").write_text(
        json.dumps({"dataset": data.metadata(), "parameter_counts": counts}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(counts, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
