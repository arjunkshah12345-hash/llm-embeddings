from ci_fixture import build_fixture
from validate_study import validate_study


def test_ci_fixture_is_a_valid_complete_study_without_training(tmp_path):
    build_fixture(tmp_path)
    result = validate_study(tmp_path)
    assert result["passed"]
    assert result["expected_run_count"] == 6
    assert result["token_tolerance_by_seed"]["1337"]["delta"] == 0
