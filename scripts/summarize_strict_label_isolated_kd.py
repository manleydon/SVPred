"""Summarize prediction-distillation runs across model seeds.

The primary protocol is the label-isolated selector. When the corresponding
less-isolated runs are present, this script also reports them side by side as
the selection-protocol sensitivity used in the supplementary material.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (41, 42, 43, 44, 45)
TASKS = {
    "CityLens Population": {
        "strict": "citylens_population_prediction_kd_controls_grid64_strictiso_seed{seed}",
        "less_isolated": "citylens_population_prediction_kd_controls_grid64_seed{seed}",
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": ("prediction_kd_shuffled", "prediction_kd_satellite_teacher"),
    },
    "CityLens Healthcare overlap": {
        "strict": "citylens_healthcare_all_prediction_kd_controls_grid64_strictiso_seed{seed}",
        "less_isolated": "citylens_healthcare_all_prediction_kd_controls_grid64_seed{seed}",
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": ("prediction_kd_shuffled", "prediction_kd_satellite_teacher"),
    },
    "Haidian Population": {
        "strict": "haidian_population_nested_spatial_localshuffle_grid64_strictiso_seed{seed}",
        "less_isolated": "haidian_population_nested_spatial_localshuffle_grid64_seed{seed}",
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": (
            "prediction_kd_global_shuffled",
            "prediction_kd_local_shuffled",
            "prediction_kd_satellite_teacher",
        ),
    },
}


def score(path: Path, model: str) -> float:
    frame = pd.read_csv(path / "pooled_summary.csv")
    row = frame.loc[frame["model"] == model, "R2"]
    if len(row) != 1:
        raise ValueError(f"{path}: expected one row for {model}, found {len(row)}")
    return float(row.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-root", type=Path, default=Path("strict_kd_runs")
    )
    parser.add_argument(
        "--less-isolated-root",
        type=Path,
        default=None,
        help=(
            "Directory holding the less-isolated selection runs. When omitted, "
            "the sensitivity columns are left empty."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("strict_kd_runs/summary")
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for task, spec in TASKS.items():
        for seed in SEEDS:
            strict_dir = args.strict_root / spec["strict"].format(seed=seed)
            strict_paired = score(strict_dir, spec["paired"])
            strict_no_privilege = score(strict_dir, spec["no_privilege"])
            strict_controls = {
                model: score(strict_dir, model) for model in spec["controls"]
            }
            strict_best_name = max(strict_controls, key=strict_controls.get)
            row: dict[str, object] = {
                "task": task,
                "seed": seed,
                "strict_paired_R2": strict_paired,
                "strict_no_privilege_R2": strict_no_privilege,
                "strict_best_control": strict_best_name,
                "strict_best_control_R2": strict_controls[strict_best_name],
                "strict_paired_minus_no_privilege": strict_paired
                - strict_no_privilege,
                "strict_paired_minus_best_control": strict_paired
                - strict_controls[strict_best_name],
            }
            if args.less_isolated_root is not None:
                less_dir = args.less_isolated_root / spec["less_isolated"].format(
                    seed=seed
                )
                less_paired = score(less_dir, spec["paired"])
                less_no_privilege = score(less_dir, spec["no_privilege"])
                less_controls = {
                    model: score(less_dir, model) for model in spec["controls"]
                }
                less_best_name = max(less_controls, key=less_controls.get)
                row.update(
                    {
                        "less_isolated_paired_R2": less_paired,
                        "less_isolated_no_privilege_R2": less_no_privilege,
                        "less_isolated_best_control": less_best_name,
                        "less_isolated_best_control_R2": less_controls[
                            less_best_name
                        ],
                        "less_isolated_paired_minus_no_privilege": less_paired
                        - less_no_privilege,
                        "less_isolated_paired_minus_best_control": less_paired
                        - less_controls[less_best_name],
                    }
                )
            rows.append(row)
            audit = pd.read_csv(strict_dir / "label_isolation_audit.csv")
            overlap_columns = [
                "selection_validation_in_teacher_crossfit_pool",
                "selection_validation_in_candidate_fit_or_earlystop",
                "outer_test_in_any_internal_set",
            ]
            audit_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    **{
                        f"max_{column}": int(audit[column].max())
                        for column in overlap_columns
                    },
                }
            )

    per_seed = pd.DataFrame(rows)
    audit_summary = pd.DataFrame(audit_rows)
    aggregate_rows = []
    metrics = [
        "strict_paired_R2",
        "strict_no_privilege_R2",
        "strict_paired_minus_no_privilege",
        "strict_paired_minus_best_control",
    ]
    if args.less_isolated_root is not None:
        metrics += [
            "less_isolated_paired_minus_no_privilege",
            "less_isolated_paired_minus_best_control",
        ]
    for task, frame in per_seed.groupby("task", sort=False):
        row: dict[str, object] = {"task": task, "seeds": len(frame)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
            row[f"{metric}_positive_seeds"] = int((frame[metric] > 0).sum())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "strict_kd_seed_comparison.csv", index=False)
    aggregate.to_csv(args.output_dir / "strict_kd_five_seed_summary.csv", index=False)
    audit_summary.to_csv(args.output_dir / "strict_kd_isolation_audit.csv", index=False)
    if audit_summary.filter(like="max_").to_numpy().max() != 0:
        raise RuntimeError("At least one strict isolation overlap is nonzero.")
    print("Strict label-isolated KD five-seed summary")
    columns = [
        "task",
        "strict_paired_R2_mean",
        "strict_paired_R2_sd",
        "strict_paired_minus_no_privilege_mean",
        "strict_paired_minus_no_privilege_positive_seeds",
        "strict_paired_minus_best_control_mean",
        "strict_paired_minus_best_control_positive_seeds",
    ]
    print(aggregate[columns].to_string(index=False))
    print("\nIsolation audit: all recorded overlap maxima are zero.")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
