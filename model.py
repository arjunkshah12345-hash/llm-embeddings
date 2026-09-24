from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from token_classes import TOKEN_CLASSES


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
    """Input/output embeddings for tied, untied, and low-rank role corrections."""

    def __init__(
        self,
        embedding_type: str,
        vocab_size: int,
        n_embd: int,
        adapter_rank: int,
        adapter_alpha: float,
    ):
        super().__init__()
        valid_types = {"tied", "untied", "partial", "partial_input", "partial_output", "capacity_control"}
        if embedding_type not in valid_types:
            raise ValueError(f"embedding_type must be one of {sorted(valid_types)}")
        if adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.embedding_type = embedding_type
        self.base_embedding_type = "tied" if embedding_type == "capacity_control" else embedding_type
        self.input_correction_enabled = embedding_type in {"partial", "partial_input"}
        self.output_correction_enabled = embedding_type in {"partial", "partial_output"}
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.adapter_rank = adapter_rank
        self.adapter_scale = adapter_alpha / adapter_rank

        if self.base_embedding_type == "untied":
            self.input_weight = nn.Parameter(torch.empty(vocab_size, n_embd))
            self.output_weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        else:
            self.shared = nn.Parameter(torch.empty(vocab_size, n_embd))
            if self.input_correction_enabled:
                self.input_a = nn.Parameter(torch.empty(vocab_size, adapter_rank))
                self.input_b = nn.Parameter(torch.empty(n_embd, adapter_rank))
            if self.output_correction_enabled:
                self.output_a = nn.Parameter(torch.empty(vocab_size, adapter_rank))
                self.output_b = nn.Parameter(torch.empty(n_embd, adapter_rank))

    def reset_parameters(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.base_embedding_type == "untied":
                nn.init.normal_(self.input_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    self.output_weight.copy_(self.input_weight)
            else:
                nn.init.normal_(self.shared, mean=0.0, std=0.02)
                if self.input_correction_enabled:
                    nn.init.normal_(self.input_a, mean=0.0, std=0.02)
                    nn.init.zeros_(self.input_b)
                if self.output_correction_enabled:
                    nn.init.normal_(self.output_a, mean=0.0, std=0.02)
                    nn.init.zeros_(self.output_b)

    def correction(self, side: str) -> torch.Tensor:
        """Materialize a correction matrix for analysis-only measurements."""
        if side not in {"input", "output"}:
            raise ValueError(f"unknown embedding side {side!r}")
        enabled = self.input_correction_enabled if side == "input" else self.output_correction_enabled
        if not enabled:
            return torch.zeros_like(self.shared)
        if side == "input":
            return self.adapter_scale * (self.input_a @ self.input_b.transpose(0, 1))
        return self.adapter_scale * (self.output_a @ self.output_b.transpose(0, 1))

    def explicit_weight(self, side: str) -> torch.Tensor:
        """Materialize the effective vocabulary-by-width matrix for analysis."""
        if side not in {"input", "output"}:
            raise ValueError(f"unknown embedding side {side!r}")
        if self.base_embedding_type == "untied":
            return self.input_weight if side == "input" else self.output_weight
        if not (self.input_correction_enabled or self.output_correction_enabled):
            return self.shared
        return self.shared + self.correction(side)

    def weight(self, side: str) -> torch.Tensor:
        """Backward-compatible alias for the analysis-only explicit matrix."""
        return self.explicit_weight(side)

    def input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up input vectors without materializing a partial correction."""
        if self.base_embedding_type == "untied":
            return F.embedding(token_ids, self.input_weight)
        shared = F.embedding(token_ids, self.shared)
        if not self.input_correction_enabled:
            return shared
        correction = F.embedding(token_ids, self.input_a) @ self.input_b.transpose(0, 1)
        return shared + self.adapter_scale * correction

    def output_logits(self, hidden: torch.Tensor, detach_weights: bool = False) -> torch.Tensor:
        """Project hidden states to logits without materializing a partial correction."""
        if self.base_embedding_type == "untied":
            weight = self.output_weight.detach() if detach_weights else self.output_weight
            return F.linear(hidden, weight)
        if not self.output_correction_enabled:
            weight = self.shared.detach() if detach_weights else self.shared
            return F.linear(hidden, weight)

        shared = self.shared.detach() if detach_weights else self.shared
        output_a = self.output_a.detach() if detach_weights else self.output_a
        output_b = self.output_b.detach() if detach_weights else self.output_b
        logits = F.linear(hidden, shared)
        low_rank_logits = (hidden @ output_b) @ output_a.transpose(0, 1)
        return logits + self.adapter_scale * low_rank_logits

    def input_parameters(self) -> list[nn.Parameter]:
        if self.base_embedding_type == "untied":
            return [self.input_weight]
        if not self.input_correction_enabled:
            return [self.shared]
        return [self.shared, self.input_a, self.input_b]

    def output_parameters(self) -> list[nn.Parameter]:
        if self.base_embedding_type == "untied":
            return [self.output_weight]
        if not self.output_correction_enabled:
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
        input_correction_compute_parameters = self.n_embd * self.adapter_rank if self.input_correction_enabled else 0
        output_correction_compute_parameters = (
            self.adapter_rank * (self.vocab_size + self.n_embd) if self.output_correction_enabled else 0
        )
        output_projection_parameters = self.vocab_size * self.n_embd
        counts = {
            "embedding_parameters": sum(p.numel() for p in self.unique_parameters()),
            "input_side_parameters": sum(p.numel() for p in self.input_parameters()),
            "output_side_parameters": sum(p.numel() for p in self.output_parameters()),
            "shared_parameters": int(self.shared.numel()) if self.base_embedding_type != "untied" else 0,
            "input_correction_parameters": 0,
            "output_correction_parameters": 0,
            "output_projection_parameters": output_projection_parameters,
            "input_correction_compute_parameters": input_correction_compute_parameters,
            "output_correction_compute_parameters": output_correction_compute_parameters,
            "embedding_compute_parameters": (
                output_projection_parameters
                + input_correction_compute_parameters
                + output_correction_compute_parameters
            ),
        }
        if self.input_correction_enabled:
            counts["input_correction_parameters"] = self.input_a.numel() + self.input_b.numel()
        if self.output_correction_enabled:
            counts["output_correction_parameters"] = self.output_a.numel() + self.output_b.numel()
        return counts

    def adapter_metrics(self) -> dict[str, float]:
        if not (self.input_correction_enabled or self.output_correction_enabled):
            return {
                "input_correction_norm": 0.0,
                "output_correction_norm": 0.0,
                "input_correction_relative_norm": 0.0,
                "output_correction_relative_norm": 0.0,
                "input_correction_effective_rank": 0.0,
                "output_correction_effective_rank": 0.0,
                "input_correction_top_singular_value": 0.0,
                "output_correction_top_singular_value": 0.0,
                "input_correction_shared_cosine": 0.0,
                "output_correction_shared_cosine": 0.0,
                "input_output_correction_cosine": 0.0,
                "input_output_left_subspace_overlap": 0.0,
                "input_output_right_subspace_overlap": 0.0,
            }

        def factor_singular_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # A @ B.T has the same non-zero singular values as the small core
            # R_A @ R_B.T from the reduced QR factorizations.
            a = a.detach().float().cpu()
            b = b.detach().float().cpu()
            _, r_a = torch.linalg.qr(a, mode="reduced")
            _, r_b = torch.linalg.qr(b, mode="reduced")
            return torch.linalg.svdvals(self.adapter_scale * (r_a @ r_b.transpose(0, 1)))

        def side_metrics(side: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
            correction = self.correction(side)
            singular_values = factor_singular_values(a, b)
            top = singular_values.max().item() if singular_values.numel() else 0.0
            threshold = top * 1e-3
            effective_rank = float((singular_values > threshold).sum().item()) if top > 0 else 0.0
            correction_norm = correction.float().norm().item()
            shared_norm = self.shared.float().norm().item()
            denominator = correction_norm * shared_norm
            shared_cosine = (
                torch.dot(correction.float().reshape(-1), self.shared.float().reshape(-1)).item() / denominator
                if denominator > 0
                else 0.0
            )
            return {
                f"{side}_correction_norm": correction_norm,
                f"{side}_correction_relative_norm": correction_norm / max(shared_norm, 1e-12),
                f"{side}_correction_effective_rank": effective_rank,
                f"{side}_correction_top_singular_value": top,
                f"{side}_correction_shared_cosine": shared_cosine,
            }

        with torch.no_grad():
            metrics = {
                "input_output_correction_cosine": 0.0,
                "input_output_left_subspace_overlap": 0.0,
                "input_output_right_subspace_overlap": 0.0,
            }
            if self.input_correction_enabled:
                metrics.update(side_metrics("input", self.input_a, self.input_b))
            if self.output_correction_enabled:
                metrics.update(side_metrics("output", self.output_a, self.output_b))
            if self.input_correction_enabled and self.output_correction_enabled:
                input_correction = self.correction("input").float()
                output_correction = self.correction("output").float()
                denominator = input_correction.norm() * output_correction.norm()
                metrics["input_output_correction_cosine"] = (
                    torch.sum(input_correction * output_correction).item() / denominator.item()
                    if denominator.item() > 0
                    else 0.0
                )

                def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
                    left_q, _ = torch.linalg.qr(left.detach().float().cpu(), mode="reduced")
                    right_q, _ = torch.linalg.qr(right.detach().float().cpu(), mode="reduced")
                    singular = torch.linalg.svdvals(left_q.transpose(0, 1) @ right_q)
                    return float((singular.pow(2).mean()).item()) if singular.numel() else 0.0

                metrics["input_output_left_subspace_overlap"] = subspace_overlap(self.input_a, self.output_a)
                metrics["input_output_right_subspace_overlap"] = subspace_overlap(self.input_b, self.output_b)
            for side in ("input", "output"):
                metrics.setdefault(f"{side}_correction_norm", 0.0)
                metrics.setdefault(f"{side}_correction_relative_norm", 0.0)
                metrics.setdefault(f"{side}_correction_effective_rank", 0.0)
                metrics.setdefault(f"{side}_correction_top_singular_value", 0.0)
                metrics.setdefault(f"{side}_correction_shared_cosine", 0.0)
            return metrics


def token_class_grad_means(prefix: str, grad: torch.Tensor | None, token_ids: torch.Tensor, class_ids: torch.Tensor) -> dict[str, float]:
    """Mean embedding-row gradient norm for tokens present in this batch, by class."""
    metrics: dict[str, float] = {}
    row_norm = None if grad is None else grad.detach().float().norm(dim=1)
    flat = token_ids.reshape(-1)
    classes = class_ids.to(flat.device)[flat] if row_norm is not None else None
    for code, name in enumerate(TOKEN_CLASSES):
        if classes is None:
            count = 0
            mean = 0.0
        else:
            mask = classes == code
            count = int(mask.sum().item())
            mean = row_norm[flat[mask]].mean().item() if count else 0.0
        metrics[f"{prefix}_{name}_count"] = float(count)
        metrics[f"{prefix}_{name}_mean"] = mean
    return metrics


def token_frequency_grad_means(
    prefix: str,
    grad: torch.Tensor | None,
    token_ids: torch.Tensor,
    frequency_ids: torch.Tensor,
    bucket_count: int = 4,
) -> dict[str, float]:
    """Mean row-gradient norms within equal-count frequency strata."""
    metrics: dict[str, float] = {}
    row_norm = None if grad is None else grad.detach().float().norm(dim=1)
    flat = token_ids.reshape(-1)
    labels = frequency_ids.to(flat.device)[flat] if row_norm is not None else None
    for bucket in range(bucket_count):
        if labels is None:
            count = 0
            mean = 0.0
        else:
            mask = labels == bucket
            count = int(mask.sum().item())
            mean = row_norm[flat[mask]].mean().item() if count else 0.0
        metrics[f"{prefix}_q{bucket}_count"] = float(count)
        metrics[f"{prefix}_q{bucket}_mean"] = mean
    return metrics


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig, embedding_type: str, seed: int = 1337):
        super().__init__()
        self.config = config
        self.embedding_type = embedding_type
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.capacity_control_width = 0
        self.capacity_adapter = None
        if embedding_type == "capacity_control":
            self.capacity_control_width = config.capacity_control_width or math.ceil(
                config.adapter_rank * (config.vocab_size + config.n_embd) / config.n_embd
            )
            self.capacity_adapter = nn.Sequential(
                nn.Linear(config.n_embd, self.capacity_control_width, bias=False),
                nn.GELU(),
                nn.Linear(self.capacity_control_width, config.n_embd, bias=False),
            )
        self.embeddings = EmbeddingSystem(
            embedding_type,
            config.vocab_size,
            config.n_embd,
            config.adapter_rank,
            config.adapter_alpha,
        )
        self._reset_transformer_parameters(seed + 1)
        self.embeddings.reset_parameters(seed + 2)
        if self.capacity_adapter is not None:
            # Start as the same function as tied; the control receives a
            # residual capacity budget without changing initialization behavior.
            nn.init.zeros_(self.capacity_adapter[-1].weight)

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

    def forward(self, idx: torch.Tensor, return_hidden: bool = False, path_ablation: str = "none"):
        if path_ablation not in {"none", "stop_input", "stop_output"}:
            raise ValueError("path_ablation must be none, stop_input, or stop_output")
        _, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(f"sequence length {length} exceeds block_size={self.config.block_size}")
        positions = torch.arange(0, length, device=idx.device)
        token_embeddings = self.embeddings.input_embeddings(idx)
        if path_ablation == "stop_input":
            token_embeddings = token_embeddings.detach()
        x = self.drop(token_embeddings + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)
        if self.capacity_adapter is not None:
            hidden = hidden + self.capacity_adapter(hidden)
        logits = self.embeddings.output_logits(hidden, detach_weights=path_ablation == "stop_output")
        if return_hidden:
            return logits, hidden
        return logits

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def embedding_gradient_metrics(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        token_class_ids: torch.Tensor | None = None,
        token_frequency_ids: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Measure input-path and output-path pressure on embedding parameters.

        The input loss detaches the output matrix; the output loss detaches the hidden
        state. Both are evaluated from the same forward-pass hidden states.
        """
        logits, hidden = self(idx, return_hidden=True)
        del logits
        input_logits = self.embeddings.output_logits(hidden, detach_weights=True)
        output_logits = self.embeddings.output_logits(hidden.detach())
        input_loss = F.cross_entropy(input_logits.reshape(-1, input_logits.size(-1)), targets.reshape(-1))
        output_loss = F.cross_entropy(output_logits.reshape(-1, output_logits.size(-1)), targets.reshape(-1))
        input_params = self.embeddings.input_parameters()
        output_params = self.embeddings.output_parameters()
        input_grads = torch.autograd.grad(input_loss, input_params, retain_graph=True, allow_unused=True)
        output_grads = torch.autograd.grad(output_loss, output_params, retain_graph=True, allow_unused=True)

        def norm(grads: Iterable[torch.Tensor | None]) -> float:
            values = [g.detach().float().pow(2).sum() for g in grads if g is not None]
            return math.sqrt(torch.stack(values).sum().item()) if values else 0.0

        def vector(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
            values = [g.detach().float().reshape(-1) for g in grads if g is not None]
            return torch.cat(values) if values else torch.empty(0, device=idx.device)

        input_vector = vector(input_grads)
        output_vector = vector(output_grads)
        vector_denominator = input_vector.norm() * output_vector.norm()
        overall_cosine = (
            torch.dot(input_vector, output_vector).item() / vector_denominator.item()
            if vector_denominator.item() > 0 and input_vector.numel() == output_vector.numel()
            else 0.0
        )

        metrics = {
            "input_side_loss": input_loss.detach().item(),
            "output_side_loss": output_loss.detach().item(),
            "input_grad_norm": norm(input_grads),
            "output_grad_norm": norm(output_grads),
            "input_output_grad_cosine": overall_cosine,
        }
        metrics["output_to_input_grad_ratio"] = metrics["output_grad_norm"] / max(metrics["input_grad_norm"], 1e-12)

        shared = getattr(self.embeddings, "shared", None)
        input_matrix = None
        output_matrix = None
        if shared is not None:
            shared_input_grad = torch.autograd.grad(input_loss, shared, retain_graph=True, allow_unused=True)[0]
            shared_output_grad = torch.autograd.grad(output_loss, shared, retain_graph=True, allow_unused=True)[0]
            input_matrix = shared_input_grad
            output_matrix = shared_output_grad
            metrics["shared_input_grad_norm"] = norm([shared_input_grad])
            metrics["shared_output_grad_norm"] = norm([shared_output_grad])
            metrics["shared_output_to_input_grad_ratio"] = metrics["shared_output_grad_norm"] / max(
                metrics["shared_input_grad_norm"], 1e-12
            )
            shared_denominator = shared_input_grad.float().norm() * shared_output_grad.float().norm()
            metrics["shared_input_output_grad_cosine"] = (
                torch.dot(shared_input_grad.float().reshape(-1), shared_output_grad.float().reshape(-1)).item()
                / shared_denominator.item()
                if shared_denominator.item() > 0
                else 0.0
            )
            if input_vector.numel() != output_vector.numel():
                # One-sided corrections have different parameter-vector sizes.
                # Their meaningful common coordinate system is the shared matrix.
                metrics["input_output_grad_cosine"] = metrics["shared_input_output_grad_cosine"]
        else:
            metrics["shared_input_grad_norm"] = 0.0
            metrics["shared_output_grad_norm"] = 0.0
            metrics["shared_output_to_input_grad_ratio"] = 0.0
            metrics["shared_input_output_grad_cosine"] = 0.0
            vocab, width = self.config.vocab_size, self.config.n_embd
            for grad in input_grads:
                if grad is not None and tuple(grad.shape) == (vocab, width):
                    input_matrix = grad
                    break
            for grad in output_grads:
                if grad is not None and tuple(grad.shape) == (vocab, width):
                    output_matrix = grad
                    break
        if token_class_ids is not None:
            metrics.update(token_class_grad_means("input_token_grad", input_matrix, idx, token_class_ids))
            metrics.update(token_class_grad_means("output_token_grad", output_matrix, targets, token_class_ids))
        if token_frequency_ids is not None:
            metrics.update(token_frequency_grad_means("input_token_freq_grad", input_matrix, idx, token_frequency_ids))
            metrics.update(token_frequency_grad_means("output_token_freq_grad", output_matrix, targets, token_frequency_ids))
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
            "capacity_control_width": self.capacity_control_width,
            "capacity_control_parameters": (
                sum(p.numel() for p in self.capacity_adapter.parameters()) if self.capacity_adapter is not None else 0
            ),
            **embedding,
        }

    def config_dict(self) -> dict:
        return {"model": asdict(self.config), "embedding_type": self.embedding_type}
