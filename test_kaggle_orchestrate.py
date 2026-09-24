from pathlib import Path

import pytest

from kaggle.orchestrate import (
    PROFILE_EXPERIMENTS,
    classify_status,
    destination_for,
    kernel_slug,
)


def test_kernel_slug_and_destination_are_deterministic(tmp_path):
    assert kernel_slug("long", 2027) == "llm-embeddings-long-50k-seed2027"
    assert destination_for(tmp_path, "long", 2027) == Path(tmp_path) / "long_seed2027"


def test_private_kernel_status_is_treated_as_missing():
    output = "Permission 'kernels.get' was denied"
    assert classify_status(1, output) == "MISSING"


def test_status_classification_covers_terminal_and_live_states():
    assert classify_status(0, 'status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    assert classify_status(0, 'status "KernelWorkerStatus.COMPLETE"') == "COMPLETE"
    assert classify_status(0, 'status "KernelWorkerStatus.ERROR"') == "FAILED"


def test_unknown_kaggle_error_is_not_silently_restarted():
    with pytest.raises(RuntimeError):
        classify_status(1, "temporary API outage")


def test_profile_manifest_is_complete():
    assert set(PROFILE_EXPERIMENTS) == {
        "primary",
        "rank",
        "long",
        "small_scale",
        "second_dataset",
        "mechanism_stop_input",
        "mechanism_stop_output",
    }
