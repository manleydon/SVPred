"""CityLens spatial test-time controls.

Protocols:
  within_city_spatial: test unseen tile blocks while retaining all cities.
  city_group:          hold out complete cities with GroupKFold.

The run produces the Satellite, parameter-matched, within-city shuffled, and
Region-associated Exact test-time conditions used as reference controls by the
other CityLens scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from run_nested_spatial_kd import (
    metrics,
    predict_student,
    predict_teacher,
    spatial_train_validation_split,
    tensor,
    train_supervised_student,
    train_teacher,
)


CONTROL_MODELS = {
    "sat_supervised",
    "sat_parameter_matched",
    "sat_shuffled_real_sv",
    "sat_real_sv_teacher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("citylens_buildheight_dinov2b14_mapillary.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("citylens_buildheight_within_city_seed42"),
    )
    parser.add_argument(
        "--protocol",
        choices=("within_city_spatial", "city_group"),
        default="within_city_spatial",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--blocks-per-city", type=int, default=10)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--target-transform",
        choices=("none", "log1p"),
        default="none",
        help="Apply before every split; Population should use log1p.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def transform_target(target: np.ndarray, transform: str) -> np.ndarray:
    target = target.astype(np.float64, copy=True)
    if transform == "none":
        return target
    if transform == "log1p":
        if np.any(target < 0):
            raise ValueError("log1p target transform requires non-negative labels.")
        return np.log1p(target)
    raise ValueError(f"Unsupported target transform: {transform}")


def city_spatial_blocks(
    city: np.ndarray,
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    blocks_per_city: int,
    seed: int,
) -> np.ndarray:
    groups = np.empty(len(city), dtype=object)
    for city_name in np.unique(city):
        positions = np.flatnonzero(city == city_name)
        coordinates = np.column_stack([tile_x[positions], tile_y[positions]])
        coordinates = StandardScaler().fit_transform(coordinates)
        n_blocks = min(blocks_per_city, len(positions))
        labels = KMeans(
            n_clusters=n_blocks,
            random_state=seed,
            n_init=20,
        ).fit_predict(coordinates)
        groups[positions] = [f"{city_name}::block_{label}" for label in labels]
    return groups.astype(str)


def shuffled_within_city(
    street_features: np.ndarray,
    city: np.ndarray,
    partitions: list[np.ndarray],
    seed: int,
) -> tuple[np.ndarray, int]:
    """Derange street features within city and within data split."""
    shuffled = street_features.copy()
    rng = np.random.default_rng(seed)
    singleton_count = 0
    for partition in partitions:
        for city_name in np.unique(city[partition]):
            indices = partition[city[partition] == city_name]
            if len(indices) < 2:
                singleton_count += len(indices)
                continue
            destination = rng.permutation(indices)
            source = np.roll(destination, 1)
            shuffled[destination] = street_features[source]
    return shuffled, singleton_count


def pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_frame = predictions[predictions["model"] == "sat_supervised"]
    teacher_frame = predictions[predictions["model"] == "sat_real_sv_teacher"]
    baseline = metrics(
        baseline_frame["y_true"].to_numpy(), baseline_frame["y_pred"].to_numpy()
    )
    teacher = metrics(
        teacher_frame["y_true"].to_numpy(), teacher_frame["y_pred"].to_numpy()
    )
    oracle_gain = teacher["R2"] - baseline["R2"]
    for model_name, frame in predictions.groupby("model"):
        result = metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        delta_r2 = result["R2"] - baseline["R2"]
        drr = (
            np.nan
            if model_name in CONTROL_MODELS or oracle_gain <= 0
            else delta_r2 / oracle_gain
        )
        rows.append(
            {
                "model": model_name,
                "N": len(frame),
                "R2": result["R2"],
                "RMSE": result["RMSE"],
                "MAE": result["MAE"],
                "Delta_R2_vs_Sat": delta_r2,
                "RMSE_reduction_vs_Sat": baseline["RMSE"] - result["RMSE"],
                "MAE_reduction_vs_Sat": baseline["MAE"] - result["MAE"],
                "Oracle_gain_R2": oracle_gain,
                "DRR": drr,
            }
        )
    return pd.DataFrame(rows)


def city_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (city_name, model_name), frame in predictions.groupby(["city", "model"]):
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()
        rows.append(
            {
                "city": city_name,
                "model": model_name,
                "N": len(frame),
                "R2": r2_score(y_true, y_pred) if len(frame) >= 2 else np.nan,
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2_reliable_N30": len(frame) >= 30,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    archive = np.load(args.features, allow_pickle=False)
    sample_id = archive["sample_id"].astype(str)
    city = archive["city"].astype(str)
    tile_x = archive["tile_x"].astype(np.int64)
    tile_y = archive["tile_y"].astype(np.int64)
    target_raw = archive["target"].astype(np.float64)
    target = transform_target(target_raw, args.target_transform)
    satellite_raw = archive["satellite_features"].astype(np.float32)
    street_raw = archive["street_features"].astype(np.float32)

    if args.protocol == "city_group":
        outer_groups = city.copy()
    else:
        outer_groups = city_spatial_blocks(
            city,
            tile_x,
            tile_y,
            args.blocks_per_city,
            args.spatial_seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": sample_id,
            "city": city,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "outer_group": outer_groups,
        }
    ).to_csv(args.output_dir / "groups.csv", index=False)

    outer_folds = min(args.outer_folds, len(np.unique(outer_groups)))
    outer_splitter = GroupKFold(n_splits=outer_folds)
    split_iterator = tqdm(
        enumerate(
            outer_splitter.split(satellite_raw, target, outer_groups), start=1
        ),
        total=outer_folds,
        desc=f"CityLens {args.protocol}",
        disable=args.no_progress,
    )
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for outer_fold, (outer_train, outer_test) in split_iterator:
        split_iterator.set_postfix(
            fold=f"{outer_fold}/{outer_folds}",
            test_cities=",".join(sorted(np.unique(city[outer_test]))),
        )
        model_seed = args.seed + outer_fold
        split_seed = args.spatial_seed + outer_fold

        satellite_scaler = StandardScaler().fit(satellite_raw[outer_train])
        street_scaler = StandardScaler().fit(street_raw[outer_train])
        satellite_scaled = satellite_scaler.transform(satellite_raw).astype(np.float32)
        street_scaled = street_scaler.transform(street_raw).astype(np.float32)
        satellite_tensor = tensor(satellite_scaled, device)
        street_tensor = tensor(street_scaled, device)

        oof_teacher_prediction = np.full(len(target), np.nan, dtype=np.float64)
        oof_baseline_prediction = np.full(len(target), np.nan, dtype=np.float64)
        oof_teacher_embedding = np.full(
            (len(target), args.hidden_dim), np.nan, dtype=np.float32
        )
        inner_groups = outer_groups[outer_train]
        inner_folds = min(args.inner_folds, len(np.unique(inner_groups)))
        inner_splitter = GroupKFold(n_splits=inner_folds)
        inner_iterator = tqdm(
            enumerate(
                inner_splitter.split(
                    outer_train, target[outer_train], inner_groups
                ),
                start=1,
            ),
            total=inner_folds,
            desc="OOF teacher signals",
            leave=False,
            disable=args.no_progress,
        )
        for inner_fold, (inner_fit_position, inner_hold_position) in inner_iterator:
            inner_pool = outer_train[inner_fit_position]
            inner_hold = outer_train[inner_hold_position]
            inner_train, inner_validation = spatial_train_validation_split(
                inner_pool,
                outer_groups,
                args.validation_fraction,
                split_seed + inner_fold,
            )
            inner_teacher = train_teacher(
                satellite_tensor,
                street_tensor,
                target,
                inner_train,
                inner_validation,
                args,
                device,
                model_seed + 10 * inner_fold,
            )
            teacher_prediction, teacher_embedding, _ = predict_teacher(
                inner_teacher,
                satellite_tensor,
                street_tensor,
                inner_hold,
            )
            oof_teacher_prediction[inner_hold] = teacher_prediction
            oof_teacher_embedding[inner_hold] = teacher_embedding

            inner_baseline = train_supervised_student(
                satellite_tensor,
                target,
                inner_train,
                inner_validation,
                args,
                device,
                model_seed + 10 * inner_fold + 1,
            )
            baseline_prediction, _ = predict_student(
                inner_baseline, satellite_tensor, inner_hold
            )
            oof_baseline_prediction[inner_hold] = baseline_prediction

        if not (
            np.isfinite(oof_teacher_prediction[outer_train]).all()
            and np.isfinite(oof_baseline_prediction[outer_train]).all()
            and np.isfinite(oof_teacher_embedding[outer_train]).all()
        ):
            raise RuntimeError("Incomplete outer-training OOF teacher signals.")

        final_train, final_validation = spatial_train_validation_split(
            outer_train,
            outer_groups,
            args.validation_fraction,
            split_seed + 99,
        )
        baseline = train_supervised_student(
            satellite_tensor,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 100,
        )
        baseline_prediction, _ = predict_student(
            baseline, satellite_tensor, outer_test
        )

        teacher = train_teacher(
            satellite_tensor,
            street_tensor,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 101,
        )
        teacher_prediction, _, teacher_gate = predict_teacher(
            teacher, satellite_tensor, street_tensor, outer_test
        )

        parameter_matched = train_teacher(
            satellite_tensor,
            satellite_tensor,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 101,
        )
        parameter_prediction, _, parameter_gate = predict_teacher(
            parameter_matched,
            satellite_tensor,
            satellite_tensor,
            outer_test,
        )

        shuffled_scaled, singleton_count = shuffled_within_city(
            street_scaled,
            city,
            [final_train, final_validation, outer_test],
            split_seed + 202,
        )
        shuffled_tensor = tensor(shuffled_scaled, device)
        shuffled_teacher = train_teacher(
            satellite_tensor,
            shuffled_tensor,
            target,
            final_train,
            final_validation,
            args,
            device,
            model_seed + 101,
        )
        shuffled_prediction, _, shuffled_gate = predict_teacher(
            shuffled_teacher,
            satellite_tensor,
            shuffled_tensor,
            outer_test,
        )

        predictions: dict[str, np.ndarray] = {
            "sat_supervised": baseline_prediction,
            "sat_parameter_matched": parameter_prediction,
            "sat_shuffled_real_sv": shuffled_prediction,
            "sat_real_sv_teacher": teacher_prediction,
        }
        gate_by_model = {model: np.nan for model in predictions}
        gate_by_model["sat_parameter_matched"] = float(np.mean(parameter_gate))
        gate_by_model["sat_shuffled_real_sv"] = float(np.mean(shuffled_gate))
        gate_by_model["sat_real_sv_teacher"] = float(np.mean(teacher_gate))

        fold_baseline = metrics(target[outer_test], baseline_prediction)
        fold_teacher = metrics(target[outer_test], teacher_prediction)
        oracle_gain = fold_teacher["R2"] - fold_baseline["R2"]
        for model_name, prediction in predictions.items():
            result = metrics(target[outer_test], prediction)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "test_cities": ";".join(sorted(np.unique(city[outer_test]))),
                    "model": model_name,
                    "N": len(outer_test),
                    "R2": result["R2"],
                    "RMSE": result["RMSE"],
                    "MAE": result["MAE"],
                    "Delta_R2_vs_Sat": result["R2"] - fold_baseline["R2"],
                    "Oracle_gain_R2": oracle_gain,
                    "gate_mean": gate_by_model[model_name],
                    "shuffled_singletons": singleton_count,
                }
            )
            for position, sample_position in enumerate(outer_test):
                prediction_rows.append(
                    {
                        "sample_id": sample_id[sample_position],
                        "city": city[sample_position],
                        "tile_x": tile_x[sample_position],
                        "tile_y": tile_y[sample_position],
                        "outer_group": outer_groups[sample_position],
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "y_true": target[sample_position],
                        "y_pred": prediction[position],
                    }
                )

    predictions_frame = pd.DataFrame(prediction_rows)
    folds_frame = pd.DataFrame(fold_rows)
    pooled_frame = pooled_summary(predictions_frame)
    cities_frame = city_summary(predictions_frame)
    predictions_frame.to_csv(args.output_dir / "outer_oof_predictions.csv", index=False)
    folds_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    cities_frame.to_csv(args.output_dir / "city_metrics.csv", index=False)

    macro = (
        cities_frame[cities_frame["R2_reliable_N30"]]
        .groupby("model", as_index=False)
        .agg(
            cities=("city", "nunique"),
            R2_macro=("R2", "mean"),
            RMSE_macro=("RMSE", "mean"),
            MAE_macro=("MAE", "mean"),
        )
    )
    macro.to_csv(args.output_dir / "city_macro_summary.csv", index=False)

    config = vars(args).copy()
    config["features"] = str(config["features"])
    config["output_dir"] = str(config["output_dir"])
    config["device_resolved"] = str(device)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("\nPooled outer-OOF summary")
    print(pooled_frame.to_string(index=False))
    print("\nMacro city summary (cities with N>=30)")
    print(macro.to_string(index=False))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
