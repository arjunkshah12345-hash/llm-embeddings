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
        escaped_condition = condition.replace("_", "\\_")
        rows.append(
            f"{escaped_condition} & {tex_int(value['total_parameters'])} & "
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
    recovery_status = final["recovery_status"].replace("_", "\\ ")
    capacity = result["by_condition"].get("capacity_control")
    paired = result.get("paired_comparisons", {})

    def delta_macro(key: str) -> str:
        value = paired.get(key, {}).get("mean_delta")
        return tex_number(value, 5)

    return "\n".join(
        [
            f"\\newcommand{{\\PrimarySeedCount}}{{{tied['run_count']}}}",
            f"\\newcommand{{\\TiedFinalLoss}}{{{tex_number(final['tied'], 5)}}}",
            f"\\newcommand{{\\PartialFinalLoss}}{{{tex_number(final['partial'], 5)}}}",
            f"\\newcommand{{\\UntiedFinalLoss}}{{{tex_number(final['untied'], 5)}}}",
            f"\\newcommand{{\\CapacityFinalLoss}}{{{tex_number(capacity.get('mean_final_val_loss') if capacity else None, 5)}}}",
            f"\\newcommand{{\\TiedFinalPPL}}{{{tex_number(tied.get('mean_final_val_perplexity'), 2)}}}",
            f"\\newcommand{{\\PartialFinalPPL}}{{{tex_number(partial.get('mean_final_val_perplexity'), 2)}}}",
            f"\\newcommand{{\\UntiedFinalPPL}}{{{tex_number(untied.get('mean_final_val_perplexity'), 2)}}}",
            f"\\newcommand{{\\PartialExtraParams}}{{{tex_int(partial['additional_parameters_vs_tied'])}}}",
            f"\\newcommand{{\\UntiedExtraParams}}{{{tex_int(untied['additional_parameters_vs_tied'])}}}",
            f"\\newcommand{{\\CapacityExtraParams}}{{{tex_int(capacity['additional_parameters_vs_tied'] if capacity else None)}}}",
            f"\\newcommand{{\\PartialVsTiedDelta}}{{{delta_macro('partial_minus_tied_final_val_loss')}}}",
            f"\\newcommand{{\\UntiedVsTiedDelta}}{{{delta_macro('untied_minus_tied_final_val_loss')}}}",
            f"\\newcommand{{\\CapacityVsTiedDelta}}{{{delta_macro('capacity_control_minus_tied_final_val_loss')}}}",
            f"\\newcommand{{\\RecoveryStatus}}{{{recovery_status}}}",
            "",
        ]
    )


def _by_study_condition(result: dict, study: str, condition: str) -> dict:
    for row in result.get("comparison", []):
        if row.get("study") == study and row.get("condition") == condition:
            return row
    return {}


def supplementary_macros(
    secondary: dict | None = None,
    mechanism: dict | None = None,
    embedding: dict | None = None,
    intervention_input: dict | None = None,
    intervention_output: dict | None = None,
) -> str:
    """Generate prose macros for every numeric claim outside the main table."""
    lines: list[str] = []

    if secondary:
        for prefix, study in (("Small", "small_scale"), ("Tiny", "tiny_shakespeare")):
            for condition, label in (("tied", "Tied"), ("partial", "Partial"), ("untied", "Untied")):
                row = _by_study_condition(secondary, study, condition)
                lines.append(f"\\newcommand{{\\{prefix}{label}Final}}{{{tex_number(row.get('mean_final_val_loss'), 5)}}}")
            pair = secondary.get("paired_comparisons", {}).get(study, {}).get("partial_minus_tied_final_val_loss", {})
            lines.extend(
                [
                    f"\\newcommand{{\\{prefix}PartialDelta}}{{{tex_number(pair.get('mean_delta'), 5)}}}",
                    f"\\newcommand{{\\{prefix}PartialDeltaLow}}{{{tex_number(pair.get('ci95', {}).get('low'), 5)}}}",
                    f"\\newcommand{{\\{prefix}PartialDeltaHigh}}{{{tex_number(pair.get('ci95', {}).get('high'), 5)}}}",
                ]
            )

    if mechanism:
        partial = mechanism.get("final", {}).get("partial", {}).get("metrics", {})
        tied = mechanism.get("final", {}).get("tied", {}).get("metrics", {})
        for name, metrics, key in (
            ("MechanismTiedRatio", tied, "output_to_input_grad_ratio"),
            ("MechanismPartialRatio", partial, "output_to_input_grad_ratio"),
            ("MechanismPartialGradCosine", partial, "input_output_grad_cosine"),
            ("MechanismPartialInputNorm", partial, "input_correction_norm"),
            ("MechanismPartialOutputNorm", partial, "output_correction_norm"),
            ("MechanismPartialCorrectionCosine", partial, "input_output_correction_cosine"),
        ):
            lines.append(f"\\newcommand{{\\{name}}}{{{tex_number(metrics.get(key, {}).get('mean'), 4)}}}")

    if embedding:
        for condition, label in (("tied", "Tied"), ("partial", "Partial")):
            metrics = embedding.get("final", {}).get(condition, {}).get("input", {}).get("metrics", {})
            for key, macro in (
                ("neighbor_cosine", f"Embedding{label}NeighborCosine"),
                ("frequency_neighbor_agreement", f"Embedding{label}NeighborAgreement"),
                ("frequency_probe_macro_accuracy", f"Embedding{label}ProbeAccuracy"),
            ):
                lines.append(f"\\newcommand{{\\{macro}}}{{{tex_number(metrics.get(key, {}).get('mean'), 3)}}}")

    for label, result in (("InputStop", intervention_input), ("OutputStop", intervention_output)):
        if result:
            study = next(iter(result.get("studies", {}).values()))
            pair = study.get("paired_comparisons", {}).get("partial_minus_tied_final_val_loss", {})
            lines.extend(
                [
                    f"\\newcommand{{\\{label}Delta}}{{{tex_number(pair.get('mean_delta'), 5)}}}",
                    f"\\newcommand{{\\{label}DeltaLow}}{{{tex_number(pair.get('ci95', {}).get('low'), 5)}}}",
                    f"\\newcommand{{\\{label}DeltaHigh}}{{{tex_number(pair.get('ci95', {}).get('high'), 5)}}}",
                ]
            )

    return "\n".join(lines) + ("\n" if lines else "")


def rank_table(result: dict) -> str:
    rows = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Rank & Extra params. & Final loss & Paired $\Delta$ & 95\% CI & Seeds \\",
        r"\midrule",
    ]
    for rank, value in sorted(result["by_rank"].items(), key=lambda item: int(item[0])):
        paired = value.get("paired_final_delta", {})
        paired_ci = paired.get("ci95", {})
        rows.append(
            f"{rank} & {tex_int(value['additional_parameters_vs_tied'])} & "
            f"{tex_number(value['mean_final_val_loss'], 5)} & "
            f"{tex_number(paired.get('mean'))} & "
            f"[{tex_number(paired_ci.get('low'))}, {tex_number(paired_ci.get('high'))}] & {value['run_count']} "
            + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def mechanism_table(result: dict) -> str:
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Condition & Out./in. grad. & Grad. cosine & Input $\\Delta_i$ norm & Output $\\Delta_o$ norm & $\\Delta_i$/$\\Delta_o$ cosine \\\\",
        r"\midrule",
    ]
    for condition in ("tied", "partial", "capacity_control", "partial_input", "partial_output"):
        entry = result.get("final", {}).get(condition)
        if not entry:
            continue
        metrics = entry["metrics"]
        values = [
            metrics.get("output_to_input_grad_ratio", {}).get("mean"),
            metrics.get("input_output_grad_cosine", {}).get("mean"),
            metrics.get("input_correction_norm", {}).get("mean"),
            metrics.get("output_correction_norm", {}).get("mean"),
            metrics.get("input_output_correction_cosine", {}).get("mean"),
        ]
        escaped_condition = condition.replace("_", "\\_")
        rows.append(
            f"{escaped_condition} & "
            + " & ".join(tex_number(value, 4) for value in values)
            + r" \\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def secondary_table(result: dict) -> str:
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Study & Condition & Params. & Extra & Final loss & Seeds \\\\",
        r"\midrule",
    ]
    for row in result.get("comparison", []):
        condition = str(row["condition"]).replace("_", "\\_")
        study = str(row["study"]).replace("_", "\\_")
        rows.append(
            f"{study} & {condition} & {tex_int(row.get('total_parameters'))} & "
            f"{tex_int(row.get('additional_parameters_vs_tied'))} & "
            f"{tex_number(row.get('mean_final_val_loss'), 5)} & {row.get('seeds', '--')} "
            + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def intervention_table(input_result: dict, output_result: dict) -> str:
    rows = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Stopped path & Paired final $\Delta$ & 95\% CI & Seeds \\",
        r"\midrule",
    ]
    for label, result in (("Input", input_result), ("Output", output_result)):
        study = next(iter(result["studies"].values()))
        value = study["paired_comparisons"]["partial_minus_tied_final_val_loss"]
        ci = value["ci95"]
        rows.append(
            f"{label} & {tex_number(value['mean_delta'], 5)} & "
            f"[{tex_number(ci['low'])}, {tex_number(ci['high'])}] & {len(value['seeds'])} "
            + r"\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--rank", default="")
    parser.add_argument("--mechanism", default="")
    parser.add_argument("--secondary", default="")
    parser.add_argument("--interventions-input", default="")
    parser.add_argument("--interventions-output", default="")
    parser.add_argument("--embedding", default="")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = read_json(Path(args.primary))
    (output_dir / "primary_table.tex").write_text(primary_table(primary))
    secondary = read_json(Path(args.secondary)) if args.secondary else None
    mechanism = read_json(Path(args.mechanism)) if args.mechanism else None
    embedding = read_json(Path(args.embedding)) if args.embedding else None
    intervention_input = read_json(Path(args.interventions_input)) if args.interventions_input else None
    intervention_output = read_json(Path(args.interventions_output)) if args.interventions_output else None
    (output_dir / "result_macros.tex").write_text(
        primary_macros(primary)
        + supplementary_macros(secondary, mechanism, embedding, intervention_input, intervention_output)
    )
    if args.rank:
        (output_dir / "rank_table.tex").write_text(rank_table(read_json(Path(args.rank))))
    if args.mechanism:
        (output_dir / "mechanism_table.tex").write_text(mechanism_table(mechanism))
    if args.secondary:
        (output_dir / "secondary_table.tex").write_text(secondary_table(secondary))
    if args.interventions_input and args.interventions_output:
        (output_dir / "intervention_table.tex").write_text(
            intervention_table(intervention_input, intervention_output)
        )


if __name__ == "__main__":
    main()
