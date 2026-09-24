import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("local_kaggle_launch", Path(__file__).parent / "kaggle" / "launch.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MECHANISM_STOP_INPUT = _MODULE.MECHANISM_STOP_INPUT
MECHANISM_STOP_OUTPUT = _MODULE.MECHANISM_STOP_OUTPUT
RUN_TEMPLATE = _MODULE.RUN_TEMPLATE
kaggle_executable = _MODULE.kaggle_executable


def test_mechanism_profiles_are_matched_and_distinct():
    assert MECHANISM_STOP_INPUT["path_ablation"] == "stop_input"
    assert MECHANISM_STOP_OUTPUT["path_ablation"] == "stop_output"
    assert MECHANISM_STOP_INPUT["steps"] == MECHANISM_STOP_OUTPUT["steps"] == 10_000
    assert MECHANISM_STOP_INPUT["seeds"] == MECHANISM_STOP_OUTPUT["seeds"]
    assert MECHANISM_STOP_INPUT["embedding_types"] == MECHANISM_STOP_OUTPUT["embedding_types"]
    assert MECHANISM_STOP_INPUT["experiment_id"] != MECHANISM_STOP_OUTPUT["experiment_id"]


def test_generated_kernel_passes_path_ablation_flags():
    assert '"--path_ablation", CONFIG.get("path_ablation", "none")' in RUN_TEMPLATE
    assert '"--ablation_start", str(CONFIG.get("ablation_start", 0))' in RUN_TEMPLATE
    assert '"--ablation_end", str(CONFIG.get("ablation_end", 0))' in RUN_TEMPLATE


def test_kaggle_executable_is_a_real_file():
    executable = Path(kaggle_executable())
    assert executable.is_file()
