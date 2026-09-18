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
