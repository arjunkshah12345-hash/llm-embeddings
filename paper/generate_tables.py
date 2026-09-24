"""Generate LaTeX tables and macros from checked-in result aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tex_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def tex_int(value: int | None) -> str:
    return "--" if value is None else f"{int(value):,}".replace(",", "{,}")


def primary_table(result: dict) -> str:
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Params. & $\Delta$ params. & Final loss & 95\% CI & Best loss & Seeds \\",
        r"\midrule",
    ]
    for condition, value in result["by_condition"].items():
        ci = value["ci95_final_val_loss"]
        ci_text = f"[{tex_number(ci['low'])}, {tex_number(ci['high'])}]"
        rows.append(
            f"{condition.replace('_', r'\\_')} & {tex_int(value['total_parameters'])} & "
            f"{tex_int(value['additional_parameters_vs_tied'])} & {tex_number(value['mean_final_val_loss'], 5)} & "
            f"{ci_text} & {tex_number(value['mean_best_val_loss'], 5)} & {value['run_count']} "
            + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def primary_macros(result: dict) -> str:
    q = result["research_question"]
    final = q["final_val_loss"]
    tied = result["by_condition"]["tied"]
    partial = result["by_condition"]["partial"]
    untied = result["by_condition"]["untied"]
    return "\n".join(
        [
            f"\\newcommand{{\\PrimarySeedCount}}{{{tied['run_count']}}}",
            f"\\newcommand{{\\TiedFinalLoss}}{{{tex_number(final['tied'], 5)}}}",
            f"\\newcommand{{\\PartialFinalLoss}}{{{tex_number(final['partial'], 5)}}}",
            f"\\newcommand{{\\UntiedFinalLoss}}{{{tex_number(final['untied'], 5)}}}",
            f"\\newcommand{{\\PartialExtraParams}}{{{tex_int(partial['additional_parameters_vs_tied'])}}}",
            f"\\newcommand{{\\UntiedExtraParams}}{{{tex_int(untied['additional_parameters_vs_tied'])}}}",
            f"\\newcommand{{\\RecoveryStatus}}{{{final['recovery_status'].replace('_', r'\\ ')}}}",
            "",
        ]
    )


def rank_table(result: dict) -> str:
    rows = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Rank & Extra params. & Final loss & 95\% CI & Seeds \\",
        r"\midrule",
    ]
    for rank, value in sorted(result["by_rank"].items(), key=lambda item: int(item[0])):
        ci = value["ci95_final_val_loss"]
        rows.append(
            f"{rank} & {tex_int(value['additional_parameters_vs_tied'])} & "
            f"{tex_number(value['mean_final_val_loss'], 5)} & "
            f"[{tex_number(ci['low'])}, {tex_number(ci['high'])}] & {value['run_count']} "
            + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--rank", default="")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = read_json(Path(args.primary))
    (output_dir / "primary_table.tex").write_text(primary_table(primary))
    (output_dir / "result_macros.tex").write_text(primary_macros(primary))
    if args.rank:
        (output_dir / "rank_table.tex").write_text(rank_table(read_json(Path(args.rank))))


if __name__ == "__main__":
    main()
