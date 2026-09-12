"""Distance-response Oracle experiment for Haidian population.

This experiment tests the scale hypothesis without changing dataset, target,
features, model capacity, or spatial folds.  Its primary estimand keeps the
exact-pair teacher fixed and replaces only its street input at outer-test time.
This isolates counterfactual sensitivity to replacement distance.  Optional
distance-specific retrained Oracles are retained as a secondary robustness
analysis, but they do not by themselves identify a distance response because
each model may learn to ignore or regularize against mismatched street input.

The replacement is deliberately performed independently inside final-training,
validation, and outer-test partitions.  No street feature crosses a partition.
When an annulus has no candidate, the closest non-self candidate to that
annulus is used and explicitly marked as a fallback.  Always inspect fallback
rates before interpreting a distance band.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from run_haidian_nested_spatial_controls import load_archive
from run_nested_spatial_kd import (
    predict_student,
    predict_teacher,
    resolve_device,
    spatial_groups,
    spatial_train_validation_split,
    tensor,
    train_supervised_student,
    train_teacher,
)


EARTH_RADIUS_M = 6_371_008.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="population")
    parser.add_argument(
        "--distance-bands-m",
        default="0-500,500-1000,1000-2000,2000-5000,5000-inf",
        help="Comma-separated replacement annuli in metres, e.g. 0-500,500-1000.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--spatial-blocks", type=int, default=25)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-seed", type=int, default=42)
    parser.add_argument("--matching-seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--skip-retrained-oracles",
        action="store_true",
        help="Run only the primary fixed-exact-teacher counterfactual curve.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_distance_bands(specification: str) -> list[tuple[float, float, str]]:
    bands: list[tuple[float, float, str]] = []
    pattern = re.compile(r"^\s*([0-9.]+)\s*-\s*([0-9.]+|inf)\s*$", re.I)
    for item in specification.split(","):
        match = pattern.match(item)
        if match is None:
            raise ValueError(f"Invalid distance band: {item!r}")
        lower = float(match.group(1))
        upper = math.inf if match.group(2).lower() == "inf" else float(match.group(2))
        if lower < 0 or upper <= lower:
            raise ValueError(f"Invalid distance interval: {item!r}")
        upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
        label = f"{lower:g}_{upper_label}m".replace(".", "p")
        bands.append((lower, upper, label))
    if not bands:
        raise ValueError("At least one distance band is required.")
    return bands


def haversine_matrix_m(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(latitude, dtype=np.float64))
    lon = np.radians(np.asarray(longitude, dtype=np.float64))
    dlat = lat[None, :] - lat[:, None]
    dlon = lon[None, :] - lon[:, None]
    value = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def distance_to_interval(distance: np.ndarray, lower: float, upper: float) -> np.ndarray:
    below = np.maximum(lower - distance, 0.0)
    above = np.zeros_like(distance) if math.isinf(upper) else np.maximum(distance - upper, 0.0)
    return below + above


def match_partition(
    partition: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    lower: float,
    upper: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source index, realized distance, and fallback flag per destination."""
    partition = np.asarray(partition, dtype=np.int64)
    if len(partition) < 2:
        raise ValueError("A replacement partition must contain at least two samples.")
    distance = haversine_matrix_m(latitude[partition], longitude[partition])
    np.fill_diagonal(distance, np.inf)
    rng = np.random.default_rng(seed)
    source_usage = np.zeros(len(partition), dtype=np.int64)
    source_local = np.full(len(partition), -1, dtype=np.int64)
    fallback = np.zeros(len(partition), dtype=bool)

    # Process constrained destinations first, then prefer the least-used source.
    valid_counts = ((distance >= lower) & (distance < upper)).sum(axis=1)
    order = np.lexsort((rng.random(len(partition)), valid_counts))
    for destination in order:
        valid = np.flatnonzero(
            (distance[destination] >= lower) & (distance[destination] < upper)
        )
        if len(valid) == 0:
            fallback[destination] = True
            penalty = distance_to_interval(distance[destination], lower, upper)
            penalty[destination] = np.inf
            best_penalty = np.nanmin(penalty)
            valid = np.flatnonzero(np.isclose(penalty, best_penalty))
        usage = source_usage[valid]
        least_used = valid[usage == usage.min()]
        chosen = int(rng.choice(least_used))
        source_local[destination] = chosen
        source_usage[chosen] += 1

    source = partition[source_local]
    realized = distance[np.arange(len(partition)), source_local]
    if np.any(source == partition) or not np.isfinite(realized).all():
        raise RuntimeError("Distance matcher produced an invalid self/non-finite match.")
    return source, realized, fallback


def make_distance_replacement(
    street: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    partitions: list[np.ndarray],
    lower: float,
    upper: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    replaced = street.copy()
    source_index = np.full(len(street), -1, dtype=np.int64)
    realized_distance = np.full(len(street), np.nan, dtype=np.float64)
    fallback = np.zeros(len(street), dtype=bool)
    for number, partition in enumerate(partitions):
        source, distance, used_fallback = match_partition(
            partition,
            latitude,
            longitude,
            lower,
            upper,
            seed + 1009 * number,
        )
        replaced[partition] = street[source]
        source_index[partition] = source
        realized_distance[partition] = distance
        fallback[partition] = used_fallback
    covered = np.concatenate(partitions)
    if np.any(source_index[covered] < 0):
        raise RuntimeError("Incomplete distance replacement.")
    return replaced, source_index, realized_distance, fallback


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = {
        model: metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        for model, frame in predictions.groupby("model", sort=False)
    }
    satellite = scores["sat_supervised"]
    exact = scores["sv_exact"]
    rows: list[dict[str, object]] = []
    for model, score in scores.items():
        frame = predictions[predictions["model"] == model]
        valid_distance = frame["replacement_distance_m"].dropna()
        has_distance = len(valid_distance) > 0
        rows.append(
            {
                "model": model,
                "N": len(frame),
                **score,
                "Delta_R2_vs_Sat": score["R2"] - satellite["R2"],
                "Delta_R2_Exact_minus_Model": exact["R2"] - score["R2"],
                "MAE_reduction_Exact_vs_Model": score["MAE"] - exact["MAE"],
                "distance_mean_m": valid_distance.mean() if has_distance else np.nan,
                "distance_p50_m": valid_distance.median() if has_distance else np.nan,
                "distance_p90_m": valid_distance.quantile(0.90) if has_distance else np.nan,
                "distance_p95_m": valid_distance.quantile(0.95) if has_distance else np.nan,
                "fallback_rate": frame["fallback"].dropna().mean() if has_distance else np.nan,
                "unique_source_ratio": frame["source_sample_id"].nunique() / len(frame)
                if has_distance
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spatial_block_bootstrap(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    exact = predictions[predictions["model"] == "sv_exact"].set_index("sample_id")
    comparison_models = [
        model
        for model in predictions["model"].unique()
        if model.startswith("sv_fixed_exact_at_distance_")
        or model.startswith("sv_distance_")
    ]
    groups = exact["spatial_group"].unique()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model in comparison_models:
        comparison = predictions[predictions["model"] == model].set_index("sample_id")
        comparison = comparison.loc[exact.index]
        deltas_r2: list[float] = []
        deltas_mae: list[float] = []
        for _ in range(repetitions):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled_positions = np.concatenate(
                [np.flatnonzero(exact["spatial_group"].to_numpy() == group) for group in sampled_groups]
            )
            y = exact["y_true"].to_numpy()[sampled_positions]
            exact_prediction = exact["y_pred"].to_numpy()[sampled_positions]
            comparison_prediction = comparison["y_pred"].to_numpy()[sampled_positions]
            deltas_r2.append(
                float(r2_score(y, exact_prediction) - r2_score(y, comparison_prediction))
            )
            deltas_mae.append(
                float(
                    mean_absolute_error(y, exact_prediction)
                    - mean_absolute_error(y, comparison_prediction)
                )
            )
        delta_r2 = np.asarray(deltas_r2)
        delta_mae = np.asarray(deltas_mae)
        observed_r2 = r2_score(exact["y_true"], exact["y_pred"]) - r2_score(
            comparison["y_true"], comparison["y_pred"]
        )
        observed_mae = mean_absolute_error(
            comparison["y_true"], comparison["y_pred"]
        ) - mean_absolute_error(exact["y_true"], exact["y_pred"])
        rows.append(
            {
                "comparison_model": model,
                "N_blocks": len(groups),
                "bootstrap_repetitions": repetitions,
                "Exact_minus_Model_R2": observed_r2,
                "R2_CI_low": np.quantile(delta_r2, 0.025),
                "R2_CI_high": np.quantile(delta_r2, 0.975),
                "R2_probability_le_zero": np.mean(delta_r2 <= 0.0),
                "MAE_reduction_Exact_vs_Model": observed_mae,
                "MAE_reduction_CI_low": np.quantile(-delta_mae, 0.025),
                "MAE_reduction_CI_high": np.quantile(-delta_mae, 0.975),
            }
        )
    return pd.DataFrame(rows)


def add_rows(
    rows: list[dict[str, object]],
    data: dict[str, np.ndarray],
    target: np.ndarray,
    indices: np.ndarray,
    spatial_group: np.ndarray,
    outer_fold: int,
    model: str,
    prediction: np.ndarray,
    source_index: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    fallback: np.ndarray | None = None,
) -> None:
    for position, index in enumerate(indices):
        source = int(source_index[index]) if source_index is not None else int(index)
        rows.append(
            {
                "sample_id": str(data["sample_id"][index]),
                "latitude": data["latitude"][index],
                "longitude": data["longitude"][index],
                "spatial_group": int(spatial_group[index]),
                "outer_fold": outer_fold,
                "model": model,
                "y_true": target[index],
                "y_pred": prediction[position],
                "source_sample_id": str(data["sample_id"][source]) if source_index is not None else "",
                "replacement_distance_m": distance[index] if distance is not None else np.nan,
                "fallback": bool(fallback[index]) if fallback is not None else np.nan,
            }
        )


def main() -> None:
    args = parse_args()
    bands = parse_distance_bands(args.distance_bands_m)
    if args.outer_folds < 2 or args.spatial_blocks < args.outer_folds:
        raise ValueError("Invalid outer-fold/spatial-block configuration.")
    if args.bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions.")
    device = resolve_device(args.device)
    data = load_archive(args.features)
    target_names = data["target_names"].astype(str).tolist()
    if args.target not in target_names:
        raise ValueError(f"Target {args.target!r} not found; available: {target_names}")
    target = data["targets"][:, target_names.index(args.target)].astype(np.float64)
    n_samples = len(target)
    all_indices = np.arange(n_samples)
    spatial_group = spatial_groups(
        data["latitude"], data["longitude"], args.spatial_blocks, args.spatial_seed
    )
    fold_assignment = np.full(n_samples, -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=args.outer_folds)
    for outer_fold, (_, test_position) in enumerate(
        splitter.split(all_indices, groups=spatial_group), start=1
    ):
        fold_assignment[test_position] = outer_fold

    prediction_rows: list[dict[str, object]] = []
    matching_rows: list[dict[str, object]] = []
    progress = tqdm(
        range(1, args.outer_folds + 1),
        desc="Haidian distance-response Oracle",
        disable=args.no_progress,
    )
    for outer_fold in progress:
        outer_test = np.flatnonzero(fold_assignment == outer_fold)
        outer_train = np.flatnonzero(fold_assignment != outer_fold)
        split_seed = args.spatial_seed + 1000 * outer_fold
        model_seed = args.seed + 1000 * outer_fold
        matching_seed = args.matching_seed + 1000 * outer_fold
        final_train, final_validation = spatial_train_validation_split(
            outer_train,
            spatial_group,
            args.validation_fraction,
            split_seed + 99,
        )
        partitions = [final_train, final_validation, outer_test]

        sat_scaler = StandardScaler().fit(data["satellite_features"][outer_train])
        street_scaler = StandardScaler().fit(data["street_features"][outer_train])
        satellite_scaled = sat_scaler.transform(data["satellite_features"]).astype(np.float32)
        street_scaled = street_scaler.transform(data["street_features"]).astype(np.float32)
        satellite = tensor(satellite_scaled, device)
        street = tensor(street_scaled, device)

        satellite_model = train_supervised_student(
            satellite,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 100,
        )
        exact_model = train_teacher(
            satellite,
            street,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 101,
        )
        add_rows(
            prediction_rows,
            data,
            target,
            outer_test,
            spatial_group,
            outer_fold,
            "sat_supervised",
            predict_student(satellite_model, satellite, outer_test)[0],
        )
        add_rows(
            prediction_rows,
            data,
            target,
            outer_test,
            spatial_group,
            outer_fold,
            "sv_exact",
            predict_teacher(exact_model, satellite, street, outer_test)[0],
        )

        for band_number, (lower, upper, label) in enumerate(bands):
            progress.set_postfix(fold=outer_fold, band=label)
            replaced, source, distance, fallback = make_distance_replacement(
                street_scaled,
                data["latitude"],
                data["longitude"],
                partitions,
                lower,
                upper,
                matching_seed + 10000 * band_number,
            )
            replacement_tensor = tensor(replaced, device)

            # Primary distance response: one exact-trained teacher is held fixed;
            # only its outer-test street input changes.
            fixed_model_name = f"sv_fixed_exact_at_distance_{label}"
            add_rows(
                prediction_rows,
                data,
                target,
                outer_test,
                spatial_group,
                outer_fold,
                fixed_model_name,
                predict_teacher(exact_model, satellite, replacement_tensor, outer_test)[0],
                source,
                distance,
                fallback,
            )
            if not args.skip_retrained_oracles:
                model = train_teacher(
                    satellite,
                    replacement_tensor,
                    target,
                    final_train,
                    final_validation,
                    args,
                    device,
                    model_seed + 101,
                )
                model_name = f"sv_distance_{label}"
                add_rows(
                    prediction_rows,
                    data,
                    target,
                    outer_test,
                    spatial_group,
                    outer_fold,
                    model_name,
                    predict_teacher(model, satellite, replacement_tensor, outer_test)[0],
                    source,
                    distance,
                    fallback,
                )
            for partition_name, partition in zip(
                ("train", "validation", "test"), partitions
            ):
                matching_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "model": fixed_model_name,
                        "partition": partition_name,
                        "N": len(partition),
                        "requested_lower_m": lower,
                        "requested_upper_m": upper,
                        "distance_mean_m": np.mean(distance[partition]),
                        "distance_p50_m": np.median(distance[partition]),
                        "distance_p90_m": np.quantile(distance[partition], 0.90),
                        "distance_p95_m": np.quantile(distance[partition], 0.95),
                        "distance_max_m": np.max(distance[partition]),
                        "fallback_rate": np.mean(fallback[partition]),
                        "unique_source_ratio": len(np.unique(source[partition])) / len(partition),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(prediction_rows)
    pooled = summarize(predictions)
    bootstrap = spatial_block_bootstrap(
        predictions, args.bootstrap_repetitions, args.seed + 900000
    )
    fold_rows: list[pd.DataFrame] = []
    for fold, frame in predictions.groupby("outer_fold"):
        fold_summary = summarize(frame)
        fold_summary.insert(0, "outer_fold", fold)
        fold_rows.append(fold_summary)

    predictions.to_csv(args.output_dir / "outer_oof_predictions.csv", index=False)
    pooled.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(
        args.output_dir / "fold_metrics.csv", index=False
    )
    pd.DataFrame(matching_rows).to_csv(
        args.output_dir / "matching_diagnostics.csv", index=False
    )
    bootstrap.to_csv(args.output_dir / "spatial_block_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": data["sample_id"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "spatial_group": spatial_group,
            "outer_fold": fold_assignment,
        }
    ).to_csv(args.output_dir / "spatial_assignment.csv", index=False)
    config = vars(args).copy()
    for key in ("features", "output_dir"):
        config[key] = str(config[key])
    config.update(
        {
            "device_resolved": str(device),
            "N_pairs": n_samples,
            "parsed_distance_bands_m": [
                {"lower": lower, "upper": upper, "label": label}
                for lower, upper, label in bands
            ],
            "outer_test_used_for_selection": False,
            "partition_safe_matching": True,
            "primary_estimand": "fixed_exact_teacher_with_test_time_distance_replacement",
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Pooled Haidian distance-response Oracle summary")
    print(pooled.to_string(index=False))
    print("\nSpatial-block bootstrap: Exact minus distance-controlled replacement")
    print(bootstrap.to_string(index=False))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
