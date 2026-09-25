from pathlib import Path

import pytest

import kaggle.orchestrate as orchestrate
from kaggle.collect import is_transient_download_error
from kaggle.orchestrate import (
    PROFILE_EXPERIMENTS,
    classify_status,
    destination_for,
    kernel_slug,
    launch_command,
)


def test_suffix_forwards_as_an_attached_option_value():
    command = launch_command("rank", "commit", "aks1321", 31415, "-e8")
    assert command[-1] == "--slug-suffix=-e8"


def test_kernel_slug_and_destination_are_deterministic(tmp_path):
    assert kernel_slug("long", 2027) == "llm-embeddings-long-50k-seed2027"
    assert destination_for(tmp_path, "long", 2027) == Path(tmp_path) / "long_seed2027"
    assert destination_for(tmp_path, "long", 2027, "-rerun") == Path(tmp_path) / "long_seed2027-rerun"


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


def test_kernel_status_retries_transient_network_failure(monkeypatch):
    responses = iter(
        [
            type("Result", (), {"returncode": 1, "stdout": "", "stderr": "NameResolutionError"})(),
            type("Result", (), {"returncode": 0, "stdout": 'status "KernelWorkerStatus.RUNNING"', "stderr": ""})(),
        ]
    )
    monkeypatch.setattr(orchestrate.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(orchestrate.time, "sleep", lambda _: None)
    assert orchestrate.kernel_status("kaggle", "owner/kernel") == "RUNNING"


def test_collection_classifies_interrupted_http_downloads_as_transient():
    assert is_transient_download_error("Connection broken: IncompleteRead(10 bytes read)")
    assert not is_transient_download_error("artifact commit mismatch")


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
