from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class EmbeddingSystem(nn.Module):
    """Input/output embeddings for tied, untied, and low-rank partially tied models."""

    def __init__(
        self,
        embedding_type: str,
        vocab_size: int,
        n_embd: int,
        adapter_rank: int,
        adapter_alpha: float,
    ):
        super().__init__()
        if embedding_type not in {"tied", "untied", "partial"}:
            raise ValueError("embedding_type must be tied, untied, or partial")
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.embedding_type = embedding_type
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.adapter_rank = adapter_rank
        self.adapter_scale = adapter_alpha / adapter_rank

        if embedding_type == "untied":
            self.input_weight = nn.Parameter(torch.empty(vocab_size, n_embd))
            self.output_weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        else:
            self.shared = nn.Parameter(torch.empty(vocab_size, n_embd))
            if embedding_type == "partial":
                self.input_a = nn.Parameter(torch.empty(vocab_size, adapter_rank))
                self.input_b = nn.Parameter(torch.empty(n_embd, adapter_rank))
                self.output_a = nn.Parameter(torch.empty(vocab_size, adapter_rank))
                self.output_b = nn.Parameter(torch.empty(n_embd, adapter_rank))

    def reset_parameters(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.embedding_type == "untied":
                nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    self.output_weight.copy_(self.input_weight)
            else:
                nn.init.normal_(self.shared, mean=0.0, std=0.02)
                if self.embedding_type == "partial":
                    nn.init.normal_(self.input_a, mean=0.0, std=0.02)
                    nn.init.zeros_(self.input_b)
                    nn.init.normal_(self.output_a, mean=0.0, std=0.02)
                    nn.init.zeros_(self.output_b)

    def correction(self, side: str) -> torch.Tensor:
        if self.embedding_type != "partial":
            return torch.zeros((), device=self.device, dtype=self.dtype)
        if side == "input":
            return self.adapter_scale * (self.input_a @ self.input_b.transpose(0, 1))
        if side == "output":
            return self.adapter_scale * (self.output_a @ self.output_b.transpose(0, 1))
        raise ValueError(f"unknown embedding side {side!r}")

    def weight(self, side: str) -> torch.Tensor:
        if side not in {"input", "output"}:
            raise ValueError(f"unknown embedding side {side!r}")
        if self.embedding_type == "untied":
            return self.input_weight if side == "input" else self.output_weight
        if self.embedding_type == "tied":
            return self.shared
        return self.shared + self.correction(side)

    def input_parameters(self) -> list[nn.Parameter]:
        if self.embedding_type == "untied":
            return [self.input_weight]
        if self.embedding_type == "tied":
            return [self.shared]
        return [self.shared, self.input_a, self.input_b]

    def output_parameters(self) -> list[nn.Parameter]:
        if self.embedding_type == "untied":
            return [self.output_weight]
        if self.embedding_type == "tied":
            return [self.shared]
        return [self.shared, self.output_a, self.output_b]

    def unique_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "embedding_parameters": sum(p.numel() for p in self.unique_parameters()),
            "input_side_parameters": sum(p.numel() for p in self.input_parameters()),
            "output_side_parameters": sum(p.numel() for p in self.output_parameters()),
            "shared_parameters": int(self.shared.numel()) if self.embedding_type != "untied" else 0,
            "input_correction_parameters": 0,
            "output_correction_parameters": 0,
        }
        if self.embedding_type == "partial":
            correction_count = self.input_a.numel() + self.input_b.numel()
            counts["input_correction_parameters"] = correction_count
            counts["output_correction_parameters"] = self.output_a.numel() + self.output_b.numel()
        return counts

    def adapter_metrics(self) -> dict[str, float]:
        if self.embedding_type != "partial":
            return {
                "input_correction_norm": 0.0,
                "output_correction_norm": 0.0,
                "input_correction_relative_norm": 0.0,
                "output_correction_relative_norm": 0.0,
            }
        with torch.no_grad():
            input_delta = self.correction("input")
            output_delta = self.correction("output")
            shared_norm = self.shared.norm().item()
            return {
                "input_correction_norm": input_delta.norm().item(),
                "output_correction_norm": output_delta.norm().item(),
                "input_correction_relative_norm": input_delta.norm().item() / max(shared_norm, 1e-12),
                "output_correction_relative_norm": output_delta.norm().item() / max(shared_norm, 1e-12),
            }


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig, embedding_type: str, seed: int = 1337):
        super().__init__()
        self.config = config
        self.embedding_type = embedding_type
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.embeddings = EmbeddingSystem(
            embedding_type,
            config.vocab_size,
            config.n_embd,
            config.adapter_rank,
            config.adapter_alpha,
        )
        self._reset_transformer_parameters(seed + 1)
        self.embeddings.reset_parameters(seed + 2)

    def _reset_transformer_parameters(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in self.modules():
                if module is self.embeddings:
                    continue
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        _, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(f"sequence length {length} exceeds block_size={self.config.block_size}")
        positions = torch.arange(0, length, device=idx.device)
        x = F.embedding(idx, self.embeddings.weight("input"))
        x = self.drop(x + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)
        logits = F.linear(hidden, self.embeddings.weight("output"))
        if return_hidden:
            return logits, hidden
        return logits

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def embedding_gradient_metrics(self, idx: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Measure input-path and output-path pressure on embedding parameters.

        The input loss detaches the output matrix; the output loss detaches the hidden
        state. Both are evaluated from the same forward-pass hidden states.
        """
        logits, hidden = self(idx, return_hidden=True)
        del logits
        output_weight = self.embeddings.weight("output")
        input_logits = F.linear(hidden, output_weight.detach())
        output_logits = F.linear(hidden.detach(), output_weight)
        input_loss = F.cross_entropy(input_logits.reshape(-1, input_logits.size(-1)), targets.reshape(-1))
        output_loss = F.cross_entropy(output_logits.reshape(-1, output_logits.size(-1)), targets.reshape(-1))
        input_params = self.embeddings.input_parameters()
        output_params = self.embeddings.output_parameters()
        input_grads = torch.autograd.grad(input_loss, input_params, retain_graph=True, allow_unused=True)
        output_grads = torch.autograd.grad(output_loss, output_params, retain_graph=True, allow_unused=True)

        def norm(grads: Iterable[torch.Tensor | None]) -> float:
            values = [g.detach().float().pow(2).sum() for g in grads if g is not None]
            return math.sqrt(torch.stack(values).sum().item()) if values else 0.0

        metrics = {
            "input_side_loss": input_loss.detach().item(),
            "output_side_loss": output_loss.detach().item(),
            "input_grad_norm": norm(input_grads),
            "output_grad_norm": norm(output_grads),
        }
        metrics["output_to_input_grad_ratio"] = metrics["output_grad_norm"] / max(metrics["input_grad_norm"], 1e-12)

        shared = getattr(self.embeddings, "shared", None)
        if shared is not None:
            shared_input_grad = torch.autograd.grad(input_loss, shared, retain_graph=True, allow_unused=True)[0]
            shared_output_grad = torch.autograd.grad(output_loss, shared, retain_graph=True, allow_unused=True)[0]
            metrics["shared_input_grad_norm"] = norm([shared_input_grad])
            metrics["shared_output_grad_norm"] = norm([shared_output_grad])
            metrics["shared_output_to_input_grad_ratio"] = metrics["shared_output_grad_norm"] / max(
                metrics["shared_input_grad_norm"], 1e-12
            )
        else:
            metrics["shared_input_grad_norm"] = 0.0
            metrics["shared_output_grad_norm"] = 0.0
            metrics["shared_output_to_input_grad_ratio"] = 0.0
        return metrics

    def combined_embedding_grad_norm(self) -> float:
        values = [p.grad.detach().float().pow(2).sum() for p in self.embeddings.unique_parameters() if p.grad is not None]
        return math.sqrt(torch.stack(values).sum().item()) if values else 0.0

    def parameter_counts(self) -> dict[str, int | str]:
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embeddings.parameter_counts()
        return {
            "embedding_type": self.embedding_type,
            "total_parameters": total,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "transformer_parameters": total - embedding["embedding_parameters"],
            **embedding,
        }

    def config_dict(self) -> dict:
        return {"model": asdict(self.config), "embedding_type": self.embedding_type}
