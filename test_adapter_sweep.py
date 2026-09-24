import json

from adapter_sweep import condition_name, summarize_condition, sweep_command, write_summary


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


def test_adapter_condition_uses_raw_partial_metrics_without_tied_baseline(tmp_path):
    condition_dir = tmp_path / "rank2_alpha8"
    condition_dir.mkdir()
    (condition_dir / "study_validation.json").write_text(json.dumps({"passed": True}))
    run_dir = condition_dir / "seed1337_partial"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {"vocab_size": 100, "n_embd": 16},
                "train": {"seed": 1337, "embedding_type": "partial"},
            }
        )
    )
    (run_dir / "parameter_counts.json").write_text(
        json.dumps(
            {
                "embedding_type": "partial",
                "total_parameters": 2000,
                "trainable_parameters": 2000,
                "embedding_parameters": 500,
                "transformer_parameters": 1500,
            }
        )
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "split": "val",
                "step": 2,
                "loss": 4.0,
                "perplexity": 54.6,
                "tokens_seen": 30,
            }
        )
        + "\n"
    )

    summary = summarize_condition(condition_dir, rank=2, alpha=8.0)

    assert summary["run_count"] == 1
    assert summary["mean_best_val_loss"] == 4.0
