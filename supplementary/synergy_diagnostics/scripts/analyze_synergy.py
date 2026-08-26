#!/usr/bin/env python3
"""Compute the MSO-Det synergy diagnostics from eight controlled runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


MODULES = ("A", "M", "U")
MODULE_NAMES = {"A": "ASWB", "M": "MSIA", "U": "UGDR"}
VARIANTS = {
    frozenset(): "Baseline",
    frozenset("A"): "ASWB",
    frozenset("M"): "MSIA",
    frozenset("U"): "UGDR",
    frozenset("AM"): "ASWB+MSIA",
    frozenset("AU"): "ASWB+UGDR",
    frozenset("MU"): "MSIA+UGDR",
    frozenset("AMU"): "Ours",
}
METRIC_INDEX = {
    "AP": 0,
    "AP75": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "factorial_epoch_metrics.csv",
        help="Per-epoch AP/AP75 trace file used by default.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="Optional directory containing the eight run records.",
    )
    parser.add_argument(
        "--export-traces",
        type=Path,
        default=None,
        help="When reading run records, export their AP/AP75 trace CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--moving-average", type=int, default=5)
    return parser.parse_args()


def load_json_lines(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def validate_run(run_name: str, rows: list[dict], args_path: Path, epochs: int) -> int:
    observed = [int(row["epoch"]) for row in rows]
    expected = list(range(epochs))
    if observed != expected:
        raise ValueError(
            f"{run_name}: expected epochs 0..{epochs - 1}, got "
            f"{observed[0]}..{observed[-1]} ({len(observed)} rows)"
        )
    run_args = json.loads(args_path.read_text())
    if "seed" not in run_args:
        raise ValueError(f"{run_name}: seed missing from {args_path}")
    return int(run_args["seed"])


def load_run_records(logs_root: Path, epochs: int) -> tuple[dict, dict[str, int]]:
    runs: dict[frozenset[str], list[dict]] = {}
    seeds: dict[str, int] = {}
    for modules, run_name in VARIANTS.items():
        run_dir = logs_root / run_name
        rows = load_json_lines(run_dir / "log.txt")
        seeds[run_name] = validate_run(
            run_name, rows, run_dir / "args.json", epochs
        )
        runs[modules] = rows
    return runs, seeds


def load_traces(
    path: Path, epochs: int
) -> tuple[dict[frozenset[str], list[dict]], dict[str, int]]:
    by_name = {run_name: modules for modules, run_name in VARIANTS.items()}
    runs: dict[frozenset[str], list[dict]] = {modules: [] for modules in VARIANTS}
    seeds: dict[str, int] = {}

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_name = row["variant"]
            if run_name not in by_name:
                raise ValueError(f"Unknown variant in {path}: {run_name}")
            modules = by_name[run_name]
            expected_label = set_label(modules)
            if row["modules"] != expected_label:
                raise ValueError(
                    f"{run_name}: expected module label {expected_label}, "
                    f"got {row['modules']}"
                )
            seed = int(row["seed"])
            if run_name in seeds and seeds[run_name] != seed:
                raise ValueError(f"{run_name}: inconsistent seeds in {path}")
            seeds[run_name] = seed
            runs[modules].append(
                {
                    "epoch": int(row["epoch"]),
                    "AP": float(row["AP"]),
                    "AP75": float(row["AP75"]),
                }
            )

    expected_epochs = list(range(epochs))
    for modules, run_name in VARIANTS.items():
        observed = [row["epoch"] for row in runs[modules]]
        if observed != expected_epochs:
            raise ValueError(
                f"{run_name}: expected epochs 0..{epochs - 1}, got {observed}"
            )
        if run_name not in seeds:
            raise ValueError(f"{run_name}: no records found in {path}")
    return runs, seeds


def metric(row: dict, name: str) -> float:
    if name in row:
        return float(row[name])
    return 100.0 * float(row["test_coco_eval_bbox"][METRIC_INDEX[name]])


def rounded(value: float, digits: int = 6) -> float:
    return round(value, digits)


def set_label(modules: frozenset[str]) -> str:
    return "+".join(MODULE_NAMES[module] for module in MODULES if module in modules) or "Baseline"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_traces(
    path: Path,
    runs: dict[frozenset[str], list[dict]],
    seeds: dict[str, int],
) -> None:
    rows = []
    for modules, run_name in VARIANTS.items():
        for record in runs[modules]:
            rows.append(
                {
                    "variant": run_name,
                    "modules": set_label(modules),
                    "epoch": int(record["epoch"]),
                    "seed": seeds[run_name],
                    "AP": metric(record, "AP"),
                    "AP75": metric(record, "AP75"),
                }
            )
    write_csv(
        path,
        ["variant", "modules", "epoch", "seed", "AP", "AP75"],
        rows,
    )


def trailing_mean(values: list[float], window: int) -> list[float]:
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(sum(values[start : index + 1]) / (index - start + 1))
    return output


def shapley_contribution(
    values: dict[frozenset[str], float], module: str
) -> float:
    contribution = 0.0
    others = tuple(candidate for candidate in MODULES if candidate != module)
    n_modules = len(MODULES)
    for size in range(len(others) + 1):
        for combination in itertools.combinations(others, size):
            context = frozenset(combination)
            weight = (
                math.factorial(size)
                * math.factorial(n_modules - size - 1)
                / math.factorial(n_modules)
            )
            contribution += weight * (
                values[context | {module}] - values[context]
            )
    return contribution


def main() -> None:
    args = parse_args()
    if args.logs_root is not None:
        runs, seeds = load_run_records(args.logs_root, args.epochs)
        if args.export_traces is not None:
            export_traces(args.export_traces, runs, seeds)
    else:
        if args.export_traces is not None:
            raise ValueError("--export-traces requires --logs-root")
        runs, seeds = load_traces(args.traces, args.epochs)

    unique_seeds = set(seeds.values())
    if len(unique_seeds) != 1:
        raise ValueError(f"Runs use different seeds: {seeds}")

    final_rows = []
    for modules, run_name in VARIANTS.items():
        final = runs[modules][-1]
        result = {
            "variant": run_name,
            "modules": set_label(modules),
            "epoch": int(final["epoch"]),
            "seed": seeds[run_name],
        }
        for name in METRIC_INDEX:
            value = metric(final, name)
            result[name] = rounded(value)
            result[f"{name}_reported_1dp"] = round(value, 1)
        final_rows.append(result)

    final_fields = ["variant", "modules", "epoch", "seed"]
    for name in METRIC_INDEX:
        final_fields.extend([name, f"{name}_reported_1dp"])
    write_csv(args.output_dir / "final_epoch_metrics.csv", final_fields, final_rows)

    values = {
        metric_name: {
            modules: metric(rows[-1], metric_name) for modules, rows in runs.items()
        }
        for metric_name in ("AP", "AP75")
    }
    reported_values = {
        metric_name: {
            modules: round(value, 1)
            for modules, value in metric_values.items()
        }
        for metric_name, metric_values in values.items()
    }

    marginal_rows = []
    for module in MODULES:
        other_modules = tuple(
            candidate for candidate in MODULES if candidate != module
        )
        for context_size in range(3):
            for combination in itertools.combinations(other_modules, context_size):
                context = frozenset(combination)
                row = {
                    "module_added": MODULE_NAMES[module],
                    "context": set_label(context),
                    "resulting_variant": set_label(context | {module}),
                }
                for metric_name in ("AP", "AP75"):
                    gain = (
                        values[metric_name][context | {module}]
                        - values[metric_name][context]
                    )
                    row[f"delta_{metric_name}"] = rounded(gain)
                    row[f"delta_{metric_name}_reported_1dp"] = rounded(
                        reported_values[metric_name][context | {module}]
                        - reported_values[metric_name][context],
                        1,
                    )
                marginal_rows.append(row)
    write_csv(
        args.output_dir / "conditional_marginal_gains.csv",
        [
            "module_added",
            "context",
            "resulting_variant",
            "delta_AP",
            "delta_AP_reported_1dp",
            "delta_AP75",
            "delta_AP75_reported_1dp",
        ],
        marginal_rows,
    )

    contribution_rows = []
    for module in MODULES:
        row = {"module": MODULE_NAMES[module]}
        for metric_name in ("AP", "AP75"):
            contribution = shapley_contribution(values[metric_name], module)
            reported_contribution = shapley_contribution(
                reported_values[metric_name], module
            )
            row[f"order_averaged_{metric_name}"] = rounded(contribution)
            row[f"order_averaged_{metric_name}_reported_1dp"] = rounded(
                reported_contribution, 1
            )
        contribution_rows.append(row)
    write_csv(
        args.output_dir / "order_averaged_contributions.csv",
        [
            "module",
            "order_averaged_AP",
            "order_averaged_AP_reported_1dp",
            "order_averaged_AP75",
            "order_averaged_AP75_reported_1dp",
        ],
        contribution_rows,
    )

    interaction_rows = []
    for first, second in itertools.combinations(MODULES, 2):
        pair = frozenset((first, second))
        row = {"interaction": f"{MODULE_NAMES[first]} x {MODULE_NAMES[second]}"}
        for metric_name in ("AP", "AP75"):
            interaction = (
                values[metric_name][pair]
                - values[metric_name][frozenset(first)]
                - values[metric_name][frozenset(second)]
                + values[metric_name][frozenset()]
            )
            row[metric_name] = rounded(interaction)
            reported_interaction = (
                reported_values[metric_name][pair]
                - reported_values[metric_name][frozenset(first)]
                - reported_values[metric_name][frozenset(second)]
                + reported_values[metric_name][frozenset()]
            )
            row[f"{metric_name}_reported_1dp"] = rounded(
                reported_interaction, 1
            )
        interaction_rows.append(row)

    all_modules = frozenset(MODULES)
    third_order = {"interaction": "ASWB x MSIA x UGDR"}
    for metric_name in ("AP", "AP75"):
        interaction = (
            values[metric_name][all_modules]
            - sum(
                values[metric_name][frozenset(pair)]
                for pair in itertools.combinations(MODULES, 2)
            )
            + sum(
                values[metric_name][frozenset(module)] for module in MODULES
            )
            - values[metric_name][frozenset()]
        )
        third_order[metric_name] = rounded(interaction)
        reported_interaction = (
            reported_values[metric_name][all_modules]
            - sum(
                reported_values[metric_name][frozenset(pair)]
                for pair in itertools.combinations(MODULES, 2)
            )
            + sum(
                reported_values[metric_name][frozenset(module)]
                for module in MODULES
            )
            - reported_values[metric_name][frozenset()]
        )
        third_order[f"{metric_name}_reported_1dp"] = rounded(
            reported_interaction, 1
        )
    interaction_rows.append(third_order)
    write_csv(
        args.output_dir / "factorial_interactions.csv",
        [
            "interaction",
            "AP",
            "AP_reported_1dp",
            "AP75",
            "AP75_reported_1dp",
        ],
        interaction_rows,
    )

    training_rows = []
    for epoch in range(args.epochs):
        row = {"epoch": epoch}
        for modules, run_name in VARIANTS.items():
            key = run_name.replace("+", "_")
            row[f"{key}_AP"] = rounded(metric(runs[modules][epoch], "AP"))
            row[f"{key}_AP75"] = rounded(metric(runs[modules][epoch], "AP75"))
        training_rows.append(row)
    training_fields = ["epoch"]
    for run_name in VARIANTS.values():
        key = run_name.replace("+", "_")
        training_fields.extend([f"{key}_AP", f"{key}_AP75"])
    write_csv(args.output_dir / "training_metrics.csv", training_fields, training_rows)

    add_last = {
        "ASWB_given_MSIA_UGDR": (frozenset("AMU"), frozenset("MU")),
        "MSIA_given_ASWB_UGDR": (frozenset("AMU"), frozenset("AU")),
        "UGDR_given_ASWB_MSIA": (frozenset("AMU"), frozenset("AM")),
    }
    trajectories: dict[str, list[float]] = {}
    for label, (with_module, context) in add_last.items():
        for metric_name in ("AP", "AP75"):
            trajectories[f"{label}_{metric_name}"] = [
                metric(runs[with_module][epoch], metric_name)
                - metric(runs[context][epoch], metric_name)
                for epoch in range(args.epochs)
            ]

    trajectory_rows = []
    smoothed = {
        key: trailing_mean(series, args.moving_average)
        for key, series in trajectories.items()
    }
    for epoch in range(args.epochs):
        row = {"epoch": epoch}
        for key, series in trajectories.items():
            row[key] = rounded(series[epoch])
            row[f"{key}_ma{args.moving_average}"] = rounded(smoothed[key][epoch])
        trajectory_rows.append(row)
    trajectory_fields = ["epoch"]
    for key in trajectories:
        trajectory_fields.extend([key, f"{key}_ma{args.moving_average}"])
    write_csv(
        args.output_dir / "training_conditional_margins.csv",
        trajectory_fields,
        trajectory_rows,
    )

    baseline = frozenset()
    exact_full_gain = values["AP"][all_modules] - values["AP"][baseline]
    exact_isolated_sum = sum(
        values["AP"][frozenset(module)] - values["AP"][baseline]
        for module in MODULES
    )
    reported_full_gain = (
        reported_values["AP"][all_modules] - reported_values["AP"][baseline]
    )
    reported_isolated_sum = sum(
        reported_values["AP"][frozenset(module)]
        - reported_values["AP"][baseline]
        for module in MODULES
    )
    summary = {
        "data_provenance": {
            "runs": list(VARIANTS.values()),
            "epochs_per_run": args.epochs,
            "seed": unique_seeds.pop(),
            "endpoint": "final epoch",
            "trace_metrics": ["AP", "AP75"],
        },
        "isolated_gain_retention_efficiency": {
            "definition": "full-model AP gain / sum of isolated-module AP gains",
            "reported_table_percent": rounded(
                100.0 * reported_full_gain / reported_isolated_sum, 1
            ),
            "unrounded_log_percent": rounded(
                100.0 * exact_full_gain / exact_isolated_sum, 1
            ),
            "rounding_note": (
                "The manuscript value uses the one-decimal AP values reported in the "
                "ablation table; the unrounded logs differ by 0.2 percentage points."
            ),
        },
        "endpoint_all_conditional_ap_gains_positive": all(
            row["delta_AP"] > 0 for row in marginal_rows
        ),
        "minimum_endpoint_conditional_ap_gain": rounded(
            min(row["delta_AP"] for row in marginal_rows)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
