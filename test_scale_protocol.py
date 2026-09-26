import json
from pathlib import Path

from config import ModelConfig
from data import (
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_REVISION,
    FINEWEB_EDU_SOURCE_FILES,
    FINEWEB_EDU_TRAIN_TOKENS,
    FINEWEB_EDU_VAL_TOKENS,
)
from model import GPTModel
from scale_lm_eval import CORE_TASKS, EXTENDED_TASKS, HARNESS_COMMIT


def test_scale_data_revision_and_shards_are_pinned():
    assert FINEWEB_EDU_DATASET == "HuggingFaceFW/fineweb-edu"
    assert FINEWEB_EDU_CONFIG == "sample-10BT"
    assert len(FINEWEB_EDU_REVISION) == 40
    assert FINEWEB_EDU_TRAIN_TOKENS == 20_480_000
    assert FINEWEB_EDU_VAL_TOKENS == 262_144
    assert len(FINEWEB_EDU_SOURCE_FILES) == 14
    assert [item["path"] for item in FINEWEB_EDU_SOURCE_FILES] == [
        f"sample/10BT/{index:03d}_00000.parquet" for index in range(14)
    ]
    assert all(len(item["oid"]) == 40 and item["size"] > 0 for item in FINEWEB_EDU_SOURCE_FILES)


def test_scale_parameter_counts_are_predeclared_and_capacity_matched():
    config = ModelConfig(
        block_size=512,
        n_layer=12,
        n_head=12,
        n_embd=768,
        adapter_rank=8,
        adapter_alpha=8,
        capacity_control_width=532,
    )
    counts = {kind: GPTModel(config, kind, 1337).parameter_counts() for kind in ("tied", "partial", "untied", "capacity_control")}
    assert counts["tied"]["total_parameters"] == 123_963_648
    assert counts["partial"]["total_parameters"] == 124_780_048
    assert counts["untied"]["total_parameters"] == 162_561_024
    assert counts["capacity_control"]["total_parameters"] == 124_780_800
    assert counts["partial"]["total_parameters"] - counts["tied"]["total_parameters"] == 816_400
    assert counts["capacity_control"]["total_parameters"] - counts["tied"]["total_parameters"] == 817_152
    assert abs(
        (counts["partial"]["total_parameters"] - counts["tied"]["total_parameters"])
        - (counts["capacity_control"]["total_parameters"] - counts["tied"]["total_parameters"])
    ) <= 1_000


def test_benchmark_suite_is_frozen_and_does_not_include_chat_tasks():
    assert HARNESS_COMMIT == "ddd67220430a2470529f25fd5c05a576ca1057a0"
    assert len(CORE_TASKS) == 10
    assert len(EXTENDED_TASKS) == 4
    assert not any("mtbench" in task or "alpaca" in task or "ifeval" in task for task in CORE_TASKS + EXTENDED_TASKS)
