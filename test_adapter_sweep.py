from adapter_sweep import condition_name, write_summary


def test_adapter_condition_names_are_filesystem_safe():
    assert condition_name(8, 8.0) == "rank8_alpha8"
    assert condition_name(2, 0.5) == "rank2_alpha0p5"


def test_adapter_summary_writes_table_and_plot(tmp_path):
    write_summary(
        tmp_path,
        [
            {
                "condition": "rank2_alpha8",
                "adapter_rank": 2,
                "adapter_alpha": 8.0,
                "run_count": 1,
                "total_parameters": 1234,
                "embedding_parameters": 900,
                "additional_parameters_vs_tied": 100,
                "mean_best_val_loss": 4.5,
                "std_best_val_loss": 0.0,
                "mean_best_val_loss_ci95": {"low": 4.5, "high": 4.5},
                "mean_final_val_loss": 4.6,
                "mean_best_val_perplexity": 90.0,
                "run_dir": "rank2_alpha8",
            }
        ],
    )

    summary = (tmp_path / "adapter_sweep_summary.md").read_text()
    assert "# Adapter sweep results" in summary
    assert "| 2 | 8 |" in summary
    assert (tmp_path / "adapter_sweep_summary.json").exists()
    assert (tmp_path / "adapter_tradeoff.png").stat().st_size > 0
