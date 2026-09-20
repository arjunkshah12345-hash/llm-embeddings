import json

import torch
import torch.nn.functional as F

from config import ModelConfig, TrainConfig
from data import TokenDataset
from model import GPTModel
from train import checkpoint_payload, save_checkpoint


def make_model(kind: str) -> GPTModel:
    config = ModelConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32, adapter_rank=4)
    return GPTModel(config, kind, seed=7)


def test_embedding_parameter_relationships():
    tied = make_model("tied")
    untied = make_model("untied")
    partial = make_model("partial")
    assert tied.parameter_counts()["embedding_parameters"] == 97 * 32
    assert untied.parameter_counts()["embedding_parameters"] == 2 * 97 * 32
    expected_partial = 97 * 32 + 2 * 4 * (97 + 32)
    assert partial.parameter_counts()["embedding_parameters"] == expected_partial
    assert tied.embeddings.weight("input") is tied.embeddings.weight("output")
    assert torch.equal(untied.embeddings.weight("input"), untied.embeddings.weight("output"))
    assert torch.equal(partial.embeddings.weight("input"), partial.embeddings.shared)
    assert torch.equal(partial.embeddings.weight("output"), partial.embeddings.shared)


def test_partial_adapter_metrics_report_effective_rank_and_alignment():
    model = make_model("partial")
    initial = model.embeddings.adapter_metrics()
    assert initial["input_correction_effective_rank"] == 0.0
    assert initial["output_correction_effective_rank"] == 0.0
    with torch.no_grad():
        model.embeddings.input_b.normal_()
        model.embeddings.output_b.normal_()
    metrics = model.embeddings.adapter_metrics()
    assert 0.0 < metrics["input_correction_effective_rank"] <= 4.0
    assert 0.0 < metrics["output_correction_effective_rank"] <= 4.0
    assert -1.0 <= metrics["input_correction_shared_cosine"] <= 1.0
    assert -1.0 <= metrics["output_correction_shared_cosine"] <= 1.0


def test_gradient_decomposition_and_forward():
    model = make_model("partial")
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))
    metrics = model.embedding_gradient_metrics(x, y)
    assert metrics["input_grad_norm"] > 0
    assert metrics["output_grad_norm"] > 0
    assert metrics["shared_input_grad_norm"] > 0
    assert metrics["shared_output_grad_norm"] > 0
    assert torch.isfinite(torch.tensor(metrics["output_to_input_grad_ratio"]))
    assert -1.0 <= metrics["input_output_grad_cosine"] <= 1.0
    assert -1.0 <= metrics["shared_input_output_grad_cosine"] <= 1.0

    model.loss(x, y).backward()
    assert model.combined_embedding_grad_norm() > 0


def test_fixed_validation_windows_are_deterministic():
    dataset = object.__new__(TokenDataset)
    dataset.tokens = {split: torch.arange(100, dtype=torch.long) for split in ("train", "val", "test")}
    x1, y1 = dataset.get_fixed_batch("val", batch_index=2, batch_size=2, block_size=8, device=torch.device("cpu"))
    x2, y2 = dataset.get_fixed_batch("val", batch_index=2, batch_size=2, block_size=8, device=torch.device("cpu"))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(y1, x1 + 1)
    assert x1.numel() == 16


def test_gradient_side_paths_match_finite_difference():
    model = make_model("tied")
    model.eval()
    x = torch.randint(0, 97, (1, 8))
    y = torch.randint(0, 97, (1, 8))
    _, hidden = model(x, return_hidden=True)
    shared = model.embeddings.shared
    output_snapshot = model.embeddings.weight("output").detach().clone()
    input_loss = F.cross_entropy(F.linear(hidden, output_snapshot).reshape(-1, 97), y.reshape(-1))
    input_grad = torch.autograd.grad(input_loss, shared)[0]

    direction = torch.zeros_like(shared)
    direction[0, 0] = 1.0
    base = shared.detach().clone()

    def input_loss_at(epsilon: float) -> float:
        with torch.no_grad():
            shared.copy_(base + epsilon * direction)
        _, perturbed_hidden = model(x, return_hidden=True)
        value = F.cross_entropy(F.linear(perturbed_hidden, output_snapshot).reshape(-1, 97), y.reshape(-1)).item()
        with torch.no_grad():
            shared.copy_(base)
        return value

    epsilon = 1e-3
    numeric_input = (input_loss_at(epsilon) - input_loss_at(-epsilon)) / (2 * epsilon)
    assert abs(input_grad[0, 0].item() - numeric_input) < 1e-3

    output_loss = F.cross_entropy(F.linear(hidden.detach(), shared).reshape(-1, 97), y.reshape(-1))
    output_grad = torch.autograd.grad(output_loss, shared)[0]

    def output_loss_at(epsilon: float) -> float:
        with torch.no_grad():
            shared.copy_(base + epsilon * direction)
        value = F.cross_entropy(F.linear(hidden.detach(), shared).reshape(-1, 97), y.reshape(-1)).item()
        with torch.no_grad():
            shared.copy_(base)
        return value

    numeric_output = (output_loss_at(epsilon) - output_loss_at(-epsilon)) / (2 * epsilon)
    assert abs(output_grad[0, 0].item() - numeric_output) < 1e-3


def test_compact_checkpoint_round_trip(tmp_path):
    model_config = ModelConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32, adapter_rank=4)
    train_config = TrainConfig(embedding_type="partial", steps=1, output_dir=str(tmp_path))
    model = GPTModel(model_config, "partial", seed=11)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, checkpoint_payload(model, model_config, train_config, torch.device("cpu"), 3, 4.5))
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = GPTModel(model_config, "partial", seed=99)
    restored.load_state_dict(loaded["model"])
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name]), name
    assert loaded["step"] == 3
    assert loaded["best_val_loss"] == 4.5
    assert loaded["kind"] == "compact"
    assert "optimizer" not in loaded


def test_optimizer_checkpoint_resume_round_trip(tmp_path):
    from train import estimate_flops, load_resume_checkpoint, optimizer_checkpoint_payload

    model_config = ModelConfig(vocab_size=97, block_size=16, n_layer=2, n_head=2, n_embd=32, adapter_rank=4)
    train_config = TrainConfig(embedding_type="partial", steps=5, output_dir=str(tmp_path))
    model = GPTModel(model_config, "partial", seed=11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))
    loss = model.loss(x, y)
    loss.backward()
    optimizer.step()
    path = tmp_path / "optimizer_last.pt"
    save_checkpoint(
        path,
        optimizer_checkpoint_payload(
            model,
            optimizer,
            model_config,
            train_config,
            torch.device("cpu"),
            step=2,
            best_val_loss=3.25,
            tokens_seen=64,
            training_elapsed=1.5,
        ),
    )
    restored = GPTModel(model_config, "partial", seed=0)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = load_resume_checkpoint(path, restored, restored_opt, torch.device("cpu"))
    assert loaded["kind"] == "optimizer"
    assert loaded["step"] == 2
    assert loaded["tokens_seen"] == 64
    assert loaded["training_wall_time_seconds"] == 1.5
    assert loaded["embedding_cumulative_update_norm"] == 0.0
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name]), name
    assert restored_opt.state_dict()["state"]


def test_flops_estimate_scales_with_tokens_and_embedding_cost():
    from train import estimate_flops

    tied = {"transformer_parameters": 1000, "embedding_parameters": 500}
    untied = {"transformer_parameters": 1000, "embedding_parameters": 1000}
    one = estimate_flops(tied, tokens=1)
    ten = estimate_flops(tied, tokens=10)
    assert one["estimated_flops_total"] == 6.0 * 1500
    assert ten["estimated_flops_total"] == 10 * one["estimated_flops_total"]
    untied_flops = estimate_flops(untied, tokens=1)
    assert untied_flops["estimated_flops_non_embedding"] == one["estimated_flops_non_embedding"]
    assert untied_flops["estimated_flops_total"] > one["estimated_flops_total"]


def test_truncate_metrics_drops_rows_at_or_after_resume_step(tmp_path):
    from train import truncate_metrics

    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 0, "split": "train", "loss": 1.0},
        {"step": 1, "split": "val", "loss": 0.9},
        {"step": 2, "split": "train", "loss": 0.8},
        {"step": 2, "split": "val", "loss": 0.85},
        {"step": 3, "split": "train", "loss": 0.7},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    removed = truncate_metrics(path, start_step=2)
    kept = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert removed == 3
    assert [row["step"] for row in kept] == [0, 1]
    assert truncate_metrics(path, start_step=2) == 0


def test_path_ablation_stops_one_embedding_role():
    model = make_model("untied")
    x = torch.randint(0, 97, (2, 8))
    y = torch.randint(0, 97, (2, 8))

    def backward(ablation: str) -> None:
        model.zero_grad(set_to_none=True)
        logits = model(x, path_ablation=ablation)
        F.cross_entropy(logits.reshape(-1, 97), y.reshape(-1)).backward()

    backward("stop_input")
    assert model.embeddings.input_weight.grad is None
    assert model.embeddings.output_weight.grad is not None
    assert model.embeddings.output_weight.grad.abs().sum() > 0

    backward("stop_output")
    assert model.embeddings.output_weight.grad is None
    assert model.embeddings.input_weight.grad is not None
    assert model.embeddings.input_weight.grad.abs().sum() > 0


def test_path_ablation_is_active_only_inside_interval():
    from train import active_path_ablation

    config = TrainConfig(path_ablation="stop_output", ablation_start=2, ablation_end=4, steps=10)
    assert active_path_ablation(1, config) == "none"
    assert active_path_ablation(2, config) == "stop_output"
    assert active_path_ablation(3, config) == "stop_output"
    assert active_path_ablation(4, config) == "none"


def test_token_class_gradient_means_follow_batch_tokens():
    model = make_model("tied")
    class_ids = torch.zeros(97, dtype=torch.long)
    class_ids[1] = 2
    class_ids[2] = 3
    x = torch.full((1, 4), 1)
    y = torch.full((1, 4), 2)
    metrics = model.embedding_gradient_metrics(x, y, class_ids)
    assert metrics["input_token_grad_common_count"] == 4
    assert metrics["output_token_grad_rare_count"] == 4
    assert metrics["output_token_grad_rare_mean"] > 0
    assert metrics["output_token_grad_common_count"] == 0
