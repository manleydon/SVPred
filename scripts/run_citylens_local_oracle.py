"""CityLens minimum-distance local street-view Oracle.

The script reuses a frozen ``run_citylens_minimal.py`` reference run.  Within
every outer fold, street views are replaced independently in final-train,
validation, and outer-test partitions.  Matching is restricted to the same
city and data-partition cell, excludes the exact pair, and minimizes the sum
of squared tile-coordinate distances under a one-to-one derangement.  A tiny
seeded jitter resolves ties; reported distances are the square roots of the
assigned costs.  A singleton cell receives a zero street-feature vector and is
flagged as a missing replacement because no no-self assignment exists.

Only the new local Oracle is trained.  Satellite, parameter-matched, global
shuffled, and exact Real-SV predictions are imported from the frozen reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from release_utils import load_reference
from run_citylens_minimal import resolve_device, transform_target
from run_nested_spatial_kd import (
    predict_teacher,
    spatial_train_validation_split,
    tensor,
    train_teacher,
)


LOCAL_MODEL = "sat_local_distance_matched_real_sv"
CONTROL_MODELS = {
    "sat_supervised",
    "sat_parameter_matched",
    "sat_shuffled_real_sv",
    "sat_real_sv_teacher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-transform", choices=("none", "log1p"), default="log1p")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pred", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-seed", type=int, default=42)
    parser.add_argument("--tile-zoom", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def minimum_distance_derangement(
    xy: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return source positions forming a minimum-distance no-self assignment."""
    n = len(xy)
    if n < 2:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.nan)
    squared = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    finite_max = float(np.max(squared)) if squared.size else 1.0
    penalty = max(1.0, finite_max) * (n + 1) * 1e6
    cost = squared.astype(np.float64, copy=True)
    np.fill_diagonal(cost, penalty)
    # Stable seeded tie breaking; far below one tile squared.
    cost += np.random.default_rng(seed).uniform(0.0, 1e-9, size=cost.shape)
    rows, cols = linear_sum_assignment(cost)
    source = np.empty(n, dtype=np.int64)
    source[rows] = cols
    if np.any(source == np.arange(n)) or len(np.unique(source)) != n:
        raise RuntimeError("Failed to construct a one-to-one no-self assignment.")
    return source, np.sqrt(squared[np.arange(n), source])


def latitude_and_meters_per_tile(tile_row: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Infer latitude from the first CityLens area coordinate (Web-Mercator row)."""
    world = float(2**zoom)
    latitude_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * tile_row / world)))
    meters_per_tile = 40075016.686 / world * np.cos(latitude_rad)
    return np.degrees(latitude_rad), meters_per_tile


def build_local_substitutes(
    street: np.ndarray,
    city: np.ndarray,
    xy: np.ndarray,
    tile_row: np.ndarray,
    partitions: dict[str, np.ndarray],
    zoom: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    local = np.zeros_like(street)
    information: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, object]] = []
    for partition_rank, (partition_name, indices) in enumerate(partitions.items()):
        indices = np.asarray(indices, dtype=np.int64)
        source_indices = np.full(len(indices), -1, dtype=np.int64)
        distance_tiles = np.full(len(indices), np.nan)
        missing = np.zeros(len(indices), dtype=bool)
        city_names = sorted(np.unique(city[indices]))
        for city_rank, city_name in enumerate(city_names):
            positions = np.flatnonzero(city[indices] == city_name)
            members = indices[positions]
            source_position, distance = minimum_distance_derangement(
                xy[members], seed + 100_003 * partition_rank + 10_007 * city_rank
            )
            singleton = source_position < 0
            if np.any(singleton):
                # A zero vector is a transparent missing-modality fallback. It
                # never borrows across city or train/validation/test boundaries.
                local[members[singleton]] = 0.0
                missing[positions[singleton]] = True
            valid = ~singleton
            sources = members[source_position[valid]]
            local[members[valid]] = street[sources]
            source_indices[positions[valid]] = sources
            distance_tiles[positions] = distance
            latitude, meters_per_tile = latitude_and_meters_per_tile(
                tile_row[members], zoom
            )
            distance_m = distance * meters_per_tile
            finite = np.isfinite(distance_m)
            diagnostics.append(
                {
                    "partition": partition_name,
                    "city": city_name,
                    "N": len(members),
                    "missing_singletons": int(singleton.sum()),
                    "latitude_mean": float(np.mean(latitude)),
                    "distance_mean_tiles": float(np.nanmean(distance)) if finite.any() else np.nan,
                    "distance_p50_tiles": float(np.nanmedian(distance)) if finite.any() else np.nan,
                    "distance_p90_tiles": float(np.nanquantile(distance, 0.90)) if finite.any() else np.nan,
                    "distance_mean_m": float(np.nanmean(distance_m)) if finite.any() else np.nan,
                    "distance_p50_m": float(np.nanmedian(distance_m)) if finite.any() else np.nan,
                    "distance_p90_m": float(np.nanquantile(distance_m, 0.90)) if finite.any() else np.nan,
                    "distance_max_m": float(np.nanmax(distance_m)) if finite.any() else np.nan,
                }
            )
        if np.any((source_indices >= 0) & (city[source_indices] != city[indices])):
            raise RuntimeError(f"Local substitute crossed a city boundary in {partition_name}.")
        if not np.all(np.isin(source_indices[source_indices >= 0], indices)):
            raise RuntimeError(f"Local substitute crossed the {partition_name} boundary.")
        information[partition_name] = {
            "indices": indices,
            "source_indices": source_indices,
            "distance_tiles": distance_tiles,
            "missing": missing,
        }
    return local, information, diagnostics


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = {
        model: score(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        for model, frame in predictions.groupby("model")
    }
    required = CONTROL_MODELS | {LOCAL_MODEL}
    missing = required - scores.keys()
    if missing:
        raise ValueError(f"Cannot summarize without models: {sorted(missing)}")
    sat = scores["sat_supervised"]
    exact = scores["sat_real_sv_teacher"]
    global_shuffled = scores["sat_shuffled_real_sv"]
    local = scores[LOCAL_MODEL]
    rows = []
    for model, result in scores.items():
        rows.append(
            {
                "model": model,
                "N": int((predictions["model"] == model).sum()),
                **result,
                "Delta_R2_vs_Sat": result["R2"] - sat["R2"],
                "Delta_R2_vs_Exact": result["R2"] - exact["R2"],
                "Full_oracle_gain_R2": exact["R2"] - sat["R2"],
                "Global_correspondence_gain_R2": exact["R2"] - global_shuffled["R2"],
                "Exact_over_local_gain_R2": exact["R2"] - local["R2"],
                "Local_transferable_gain_R2": local["R2"] - sat["R2"],
                "Local_value_retention": (
                    (local["R2"] - sat["R2"]) / (exact["R2"] - sat["R2"])
                    if exact["R2"] > sat["R2"]
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def city_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city_name, model), frame in predictions.groupby(["city", "model"]):
        result = score(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()) if len(frame) >= 2 else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
        rows.append({"city": city_name, "model": model, "N": len(frame), **result, "R2_reliable_N30": len(frame) >= 30})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    archive = np.load(args.features, allow_pickle=False)
    sample_id = archive["sample_id"].astype(str)
    city = archive["city"].astype(str)
    tile_x = archive["tile_x"].astype(np.int64)
    tile_y = archive["tile_y"].astype(np.int64)
    target = transform_target(archive["target"].astype(np.float64), args.target_transform)
    satellite_raw = archive["satellite_features"].astype(np.float32)
    street_raw = archive["street_features"].astype(np.float32)
    reference, fold_assignment, outer_groups = load_reference(
        args.reference_dir, sample_id, city, target, args.spatial_seed
    )
    reference = reference[reference["model"].isin(CONTROL_MODELS)].copy()
    xy = np.column_stack([tile_x, tile_y]).astype(np.float64)
    local_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    folds = sorted(np.unique(fold_assignment))
    progress = tqdm(folds, desc="CityLens local Oracle", disable=args.no_progress)
    for outer_fold in progress:
        outer_test = np.flatnonzero(fold_assignment == outer_fold)
        outer_train = np.flatnonzero(fold_assignment != outer_fold)
        split_seed = args.spatial_seed + int(outer_fold)
        model_seed = args.seed + int(outer_fold)
        final_train, final_validation = spatial_train_validation_split(
            outer_train, outer_groups, args.validation_fraction, split_seed + 99
        )
        progress.set_postfix(fold=int(outer_fold), test=len(outer_test))

        sat_scaler = StandardScaler().fit(satellite_raw[outer_train])
        street_scaler = StandardScaler().fit(street_raw[outer_train])
        satellite_scaled = sat_scaler.transform(satellite_raw).astype(np.float32)
        street_scaled = street_scaler.transform(street_raw).astype(np.float32)
        local_scaled, information, diagnostics = build_local_substitutes(
            street_scaled,
            city,
            xy,
            tile_x,
            {"train": final_train, "validation": final_validation, "test": outer_test},
            args.tile_zoom,
            split_seed + 500_202,
        )
        for row in diagnostics:
            row["outer_fold"] = int(outer_fold)
            diagnostic_rows.append(row)

        satellite_tensor = tensor(satellite_scaled, device)
        local_tensor = tensor(local_scaled, device)
        local_teacher = train_teacher(
            satellite_tensor,
            local_tensor,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 101,
        )
        prediction, _, gate = predict_teacher(
            local_teacher, satellite_tensor, local_tensor, outer_test
        )
        test_info = information["test"]
        for position, index in enumerate(outer_test):
            source_index = int(test_info["source_indices"][position])
            latitude, meters_per_tile = latitude_and_meters_per_tile(
                np.asarray([tile_x[index]], dtype=float), args.tile_zoom
            )
            distance_tiles = float(test_info["distance_tiles"][position])
            local_rows.append(
                {
                    "sample_id": sample_id[index],
                    "city": city[index],
                    "tile_x": tile_x[index],
                    "tile_y": tile_y[index],
                    "outer_group": outer_groups[index],
                    "outer_fold": int(outer_fold),
                    "model": LOCAL_MODEL,
                    "y_true": target[index],
                    "y_pred": float(prediction[position]),
                    "gate": float(np.asarray(gate[position]).mean()),
                    "local_source_id": sample_id[source_index] if source_index >= 0 else "",
                    "local_source_city": city[source_index] if source_index >= 0 else "",
                    "local_distance_tiles": distance_tiles,
                    "local_distance_m": distance_tiles * float(meters_per_tile[0]),
                    "local_missing": bool(test_info["missing"][position]),
                    "latitude_inferred": float(latitude[0]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_predictions = pd.DataFrame(local_rows)
    common = ["sample_id", "city", "tile_x", "tile_y", "outer_group", "outer_fold", "model", "y_true", "y_pred"]
    predictions = pd.concat([reference[common], local_predictions], ignore_index=True, sort=False)
    pooled = summarize(predictions)
    fold_tables = []
    for fold, frame in predictions.groupby("outer_fold"):
        fold_table = summarize(frame)
        fold_table.insert(0, "outer_fold", int(fold))
        fold_tables.append(fold_table)
    per_city = city_metrics(predictions)
    reliable = per_city[per_city["R2_reliable_N30"]]
    macro = reliable.groupby("model").agg(
        cities=("city", "nunique"),
        R2_macro=("R2", "mean"),
        RMSE_macro=("RMSE", "mean"),
        MAE_macro=("MAE", "mean"),
    ).reset_index()

    predictions.to_csv(args.output_dir / "outer_oof_predictions.csv", index=False)
    local_predictions.to_csv(args.output_dir / "local_oof_predictions.csv", index=False)
    pooled.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    per_city.to_csv(args.output_dir / "city_metrics.csv", index=False)
    macro.to_csv(args.output_dir / "city_macro_summary.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(args.output_dir / "local_matching_diagnostics.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "device_resolved": str(device),
            "N_pairs": len(target),
            "matching": "minimum-cost one-to-one no-self assignment",
            "matching_uses_target": False,
            "matching_crosses_city": False,
            "matching_crosses_partition": False,
            "tile_coordinate_note": "CityLens area first coordinate treated as Web-Mercator row at --tile-zoom for meter conversion",
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("CityLens local/distance-matched Oracle summary")
    print(pooled.to_string(index=False))
    print("\nMacro city summary (cities with N>=30)")
    print(macro.to_string(index=False))
    print("\nLocal test-distance summary")
    print(
        local_predictions.groupby("city")["local_distance_m"]
        .agg(["count", "mean", "median", "max"])
        .to_string()
    )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
