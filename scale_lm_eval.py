"""Run pinned lm-evaluation-harness tasks against a Study 3 checkpoint.

The harness is used as the task/dataset/metric implementation.  This small
adapter only exposes the repository's GPTModel through the documented custom
``LM`` interface; it does not implement benchmark scoring rules itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
import torch

from config import ModelConfig
from model import GPTModel


HARNESS_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
PREFIX_TOKEN_ID = 50256
CORE_TASKS = (
    "lambada_open",
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "sciq",
    "openbookqa",
    "boolq",
    "commonsense_qa",
)
EXTENDED_TASKS = ("mmlu", "truthfulqa_mc1", "triviaqa", "gsm8k")
HARD_TASKS = ("mmlu_pro", "bbh", "gpqa", "musr", "math")


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPTModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = ModelConfig(**config["model"])
    train_config = config["train"]
    model = GPTModel(model_config, train_config["embedding_type"], int(train_config["seed"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {"model": config["model"], "train": train_config, "step": checkpoint.get("step")}


def build_study_lm(model: GPTModel, device: torch.device, batch_size: int = 1):
        """Build an lm-eval ``LM`` wrapper around a repository checkpoint."""
        from lm_eval.api.model import LM

        class _StudyLM(LM):
            def __init__(self):
                super().__init__()
                self.model = model
                self._device = device
                self.encoder = tiktoken.get_encoding("gpt2")
                self._batch_size = batch_size
                self.max_length = model.config.block_size
                self.vocab_size = self.encoder.n_vocab
                self.prefix_token_id = PREFIX_TOKEN_ID

            @property
            def batch_size(self):
                return self._batch_size

            @property
            def max_batch_size(self):
                return self._batch_size

            def tok_encode(self, string: str, add_special_tokens: bool = True):
                del add_special_tokens
                return self.encoder.encode(string, allowed_special=set())

            def tok_decode(self, tokens):
                return self.encoder.decode(tokens)

            @torch.no_grad()
            def _score_range(self, tokens: list[int], target_start: int, target_end: int) -> tuple[float, bool]:
                """Score target token positions with up to max_length context."""
                if target_start < 1 or target_end > len(tokens) or target_start >= target_end:
                    return 0.0, True
                total = 0.0
                greedy = True
                cursor = target_start
                while cursor < target_end:
                    chunk_end = min(target_end, cursor + self.max_length - 1)
                    input_start = max(0, chunk_end - self.max_length)
                    input_tokens = tokens[input_start:chunk_end]
                    x = torch.tensor(input_tokens, dtype=torch.long, device=self.device)[None, :]
                    logits = self.model(x)[0]
                    first = max(cursor, input_start + 1)
                    positions = torch.arange(first, chunk_end, device=self.device)
                    log_probs = torch.log_softmax(logits[positions - input_start - 1], dim=-1)
                    target = torch.tensor([tokens[pos] for pos in positions.tolist()], dtype=torch.long, device=self.device)
                    chosen = log_probs.gather(1, target[:, None]).squeeze(1)
                    total += float(chosen.sum().item())
                    greedy = greedy and bool((log_probs.argmax(dim=-1) == target).all().item())
                    cursor = chunk_end
                return total, greedy

            def _pair_tokens(self, context: str, continuation: str) -> tuple[list[int], int, int]:
                if context:
                    context_tokens = self.encoder.encode(context, allowed_special=set())
                    whole_tokens = self.encoder.encode(context + continuation, allowed_special=set())
                    target_start = len(context_tokens)
                    return whole_tokens, target_start, len(whole_tokens)
                continuation_tokens = self.encoder.encode(continuation, allowed_special=set())
                whole_tokens = [self.prefix_token_id] + continuation_tokens
                return whole_tokens, 1, len(whole_tokens)

            def loglikelihood(self, requests):
                outputs = []
                for request in requests:
                    context, continuation = request.args
                    tokens, start, end = self._pair_tokens(context, continuation)
                    outputs.append(self._score_range(tokens, start, end))
                return outputs

            def loglikelihood_rolling(self, requests):
                outputs = []
                for request in requests:
                    text = request.args[0]
                    tokens = [self.prefix_token_id] + self.encoder.encode(text, allowed_special=set())
                    score, _ = self._score_range(tokens, 1, len(tokens))
                    outputs.append(score)
                return outputs

            def generate_until(self, requests):
                outputs = []
                for request in requests:
                    context, gen_kwargs = request.args
                    until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
                    max_gen = int(gen_kwargs.get("max_gen_toks", 32)) if isinstance(gen_kwargs, dict) else 32
                    tokens = self.encoder.encode(context, allowed_special=set())
                    if not tokens:
                        tokens = [self.prefix_token_id]
                    generated: list[int] = []
                    for _ in range(max_gen):
                        window = tokens[-self.max_length :]
                        x = torch.tensor(window, dtype=torch.long, device=self.device)[None, :]
                        next_token = int(self.model(x)[0, -1].argmax().item())
                        generated.append(next_token)
                        tokens.append(next_token)
                        text = self.encoder.decode(generated)
                        if any(text.endswith(stop) for stop in until):
                            break
                    text = self.encoder.decode(generated)
                    for stop in until:
                        if stop in text:
                            text = text.split(stop, 1)[0]
                    outputs.append(text)
                return outputs

        return _StudyLM()


def json_default(value):
    try:
        return value.item()
    except AttributeError:
        return str(value)


def run_suite(checkpoint: Path, output: Path, suite: str, device_name: str, batch_size: int) -> dict:
    import lm_eval

    device = torch.device(device_name)
    model, checkpoint_meta = load_checkpoint(checkpoint, device)
    task_names = {
        "core": CORE_TASKS,
        "extended": EXTENDED_TASKS,
        "hard": HARD_TASKS,
    }[suite]
    lm = build_study_lm(model, device, batch_size)
    results = {}
    for task in task_names:
        try:
            payload = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=0,
                batch_size=batch_size,
                log_samples=False,
                verbosity="ERROR",
            )
            results[task] = {
                "status": "complete",
                "result": payload.get("results", {}).get(task),
                "config": payload.get("configs", {}).get(task),
            }
        except Exception as exc:  # retain an inspectable failure instead of dropping a task
            results[task] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "harness": {"name": "lm-evaluation-harness", "commit": HARNESS_COMMIT},
        "suite": suite,
        "tasks": list(task_names),
        "num_fewshot": 0,
        "batch_size": batch_size,
        "precision": "float32",
        "checkpoint": str(checkpoint.name),
        "checkpoint_metadata": checkpoint_meta,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--suite", choices=["core", "extended", "hard", "all"], default="core")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suites = ("core", "extended", "hard") if args.suite == "all" else (args.suite,)
    for suite in suites:
        print(f"running lm-eval suite={suite}", flush=True)
        run_suite(
            args.checkpoint,
            args.output_dir / f"lm_eval_{suite}.json",
            suite,
            args.device,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
