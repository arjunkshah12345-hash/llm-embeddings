import torch

from config import ModelConfig
from model import GPTModel


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
    assert torch.equal(partial.embeddings.weight("input"), partial.embeddings.weight("output"))


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
