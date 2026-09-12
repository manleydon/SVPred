"""Grouped bootstrap inference for strict label-isolated KD outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (41, 42, 43, 44, 45)
TASKS = {
    "CityLens Population": {
        "directory": "citylens_population_prediction_kd_controls_grid64_strictiso_seed{seed}",
        "groups": ("outer_group", "city"),
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": ("prediction_kd_shuffled", "prediction_kd_satellite_teacher"),
    },
    "CityLens Healthcare overlap": {
        "directory": "citylens_healthcare_all_prediction_kd_controls_grid64_strictiso_seed{seed}",
        "groups": ("outer_group", "city"),
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": ("prediction_kd_shuffled", "prediction_kd_satellite_teacher"),
    },
    "Haidian Population": {
        "directory": "haidian_population_nested_spatial_localshuffle_grid64_strictiso_seed{seed}",
        "groups": ("spatial_group",),
        "paired": "prediction_kd_paired",
        "no_privilege": "prediction_kd_no_privilege",
        "controls": (
            "prediction_kd_global_shuffled",
            "prediction_kd_local_shuffled",
            "prediction_kd_satellite_teacher",
        ),
    },
}


def prepare(path: Path, models: tuple[str, ...], group: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "outer_oof_predictions.csv")
    required = {"sample_id", group, "model", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    subset = frame[frame["model"].isin(models)].copy()
    if set(subset["model"].unique()) != set(models):
        raise ValueError(f"{path}: incomplete model set")
    keys = ["sample_id", group]
    target = subset.groupby(keys, observed=True)["y_true"].first()
    spread = subset.groupby(keys, observed=True)["y_true"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (spread > 1e-8).any():
        raise ValueError(f"{path}: targets differ across model rows")
    wide = subset.pivot(index=keys, columns="model", values="y_pred")
    if wide.isna().any().any():
        raise ValueError(f"{path}: incomplete paired predictions")
    return wide.join(target).reset_index()


def contrasts(frame: pd.DataFrame, paired: str, no_privilege: str, controls: tuple[str, ...]) -> tuple[float, float]:
    truth = frame["y_true"].to_numpy()
    denominator = float(np.square(truth - truth.mean()).sum())
    if denominator <= 0:
        return np.nan, np.nan
    sse_paired = float(np.square(truth - frame[paired].to_numpy()).sum())
    sse_no_privilege = float(
        np.square(truth - frame[no_privilege].to_numpy()).sum()
    )
    control_sse = [
        float(np.square(truth - frame[model].to_numpy()).sum())
        for model in controls
    ]
    return (
        (sse_no_privilege - sse_paired) / denominator,
        (min(control_sse) - sse_paired) / denominator,
    )


def bootstrap(
    frame: pd.DataFrame,
    group: str,
    paired: str,
    no_privilege: str,
    controls: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    clusters = [part for _, part in frame.groupby(group, sort=False)]
    rng = np.random.default_rng(seed)
    no_privilege_values = np.empty(repetitions, dtype=np.float64)
    best_control_values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        resample = pd.concat([clusters[index] for index in sampled], ignore_index=True)
        no_privilege_values[repetition], best_control_values[repetition] = contrasts(
            resample, paired, no_privilege, controls
        )
    return no_privilege_values, best_control_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, default=Path("strict_kd_runs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("strict_kd_runs/summary")
    )
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []

    for task_index, (task, spec) in enumerate(TASKS.items()):
        models = (spec["paired"], spec["no_privilege"], *spec["controls"])
        for model_seed in SEEDS:
            path = args.strict_root / spec["directory"].format(seed=model_seed)
            for group_index, group in enumerate(spec["groups"]):
                frame = prepare(path, models, group)
                observed_no_privilege, observed_best_control = contrasts(
                    frame, spec["paired"], spec["no_privilege"], spec["controls"]
                )
                boot_no_privilege, boot_best_control = bootstrap(
                    frame,
                    group,
                    spec["paired"],
                    spec["no_privilege"],
                    spec["controls"],
                    args.repetitions,
                    args.seed + 10000 * task_index + 100 * model_seed + group_index,
                )
                for contrast_name, observed, values in (
                    ("Paired-NoPrivilege", observed_no_privilege, boot_no_privilege),
                    ("Paired-BestControl", observed_best_control, boot_best_control),
                ):
                    lower, upper = np.quantile(values[np.isfinite(values)], [0.025, 0.975])
                    rows.append(
                        {
                            "task": task,
                            "model_seed": model_seed,
                            "resampling_unit": group,
                            "contrast": contrast_name,
                            "observed_delta_R2": observed,
                            "CI95_lower": float(lower),
                            "CI95_upper": float(upper),
                            "CI_excludes_zero": bool(lower > 0 or upper < 0),
                            "positive": bool(observed > 0),
                            "repetitions": args.repetitions,
                        }
                    )

    intervals = pd.DataFrame(rows)
    counts = (
        intervals.groupby(["task", "resampling_unit", "contrast"], as_index=False)
        .agg(
            positive_seeds=("positive", "sum"),
            CI_excludes_zero_seeds=("CI_excludes_zero", "sum"),
            mean_delta_R2=("observed_delta_R2", "mean"),
            sd_delta_R2=("observed_delta_R2", "std"),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.output_dir / "strict_kd_grouped_bootstrap_intervals.csv", index=False)
    counts.to_csv(args.output_dir / "strict_kd_grouped_bootstrap_summary.csv", index=False)
    print(counts.to_string(index=False))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
