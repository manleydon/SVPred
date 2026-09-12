"""Controlled comparisons for deployable CityLens prediction distillation.

Three satellite-only students share architecture, initialization, spatial
splits and KD-weight selection.  They differ only in their cross-fitted soft
targets: paired Real-SV teacher, within-city shuffled Real-SV teacher, or an
independently trained satellite-only teacher.  Outer-test inference uses only
satellite features for every student.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from release_utils import load_reference, parse_number_list
from run_citylens_minimal import shuffled_within_city, transform_target
from run_nested_spatial_kd import (
    predict_student,
    predict_teacher,
    select_kd_student_by_spatial_validation,
    select_kd_student_with_isolated_validation,
    spatial_train_validation_split,
    tensor,
    train_supervised_student,
    train_teacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-transform", choices=("none", "log1p"), default="none"
    )
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pred", type=float, default=0.5)
    parser.add_argument(
        "--kd-weight-grid",
        default="0,0.25,0.5,1,2,4,8,16,32,64",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=(
            "Independent seed for within-city shuffled-street assignments. "
            "Defaults to --spatial-seed to preserve previous behavior."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--strict-label-isolation",
        action="store_true",
        help=(
            "Split final selection train/validation before teacher cross-fitting; "
            "generate soft targets only within selection train; and use a second "
            "selection-train-only split for candidate early stopping. Without "
            "this flag the less-isolated selection protocol is used."
        ),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = {
        model: prediction_metrics(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
        )
        for model, frame in predictions.groupby("model")
    }
    sat = scores["sat_supervised"]
    no_privilege = scores["prediction_kd_no_privilege"]
    paired_teacher = scores["sat_real_sv_teacher"]
    shuffled_teacher = scores["sat_shuffled_real_sv"]
    full_oracle_gain = paired_teacher["R2"] - sat["R2"]
    correspondence_gain = paired_teacher["R2"] - shuffled_teacher["R2"]
    rows = []
    for model, result in scores.items():
        drr = np.nan
        correspondence_recovery = np.nan
        if model.startswith("prediction_kd_") and full_oracle_gain > 0:
            drr = (result["R2"] - no_privilege["R2"]) / full_oracle_gain
        if model == "prediction_kd_paired" and correspondence_gain > 0:
            correspondence_recovery = (
                result["R2"] - scores["prediction_kd_shuffled"]["R2"]
            ) / correspondence_gain
        rows.append(
            {
                "model": model,
                "N": int((predictions["model"] == model).sum()),
                **result,
                "Delta_R2_vs_Sat": result["R2"] - sat["R2"],
                "Delta_R2_vs_no_privilege": result["R2"] - no_privilege["R2"],
                "Delta_R2_vs_shuffled_KD": (
                    result["R2"] - scores["prediction_kd_shuffled"]["R2"]
                    if model.startswith("prediction_kd_")
                    else np.nan
                ),
                "MAE_reduction_vs_no_privilege": (
                    no_privilege["MAE"] - result["MAE"]
                ),
                "Full_oracle_gain_R2": full_oracle_gain,
                "Correspondence_oracle_gain_R2": correspondence_gain,
                "Full_DRR": drr,
                "Correspondence_recovery_ratio": correspondence_recovery,
            }
        )
    return pd.DataFrame(rows)


def city_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, model), frame in predictions.groupby(["city", "model"]):
        result = prediction_metrics(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
        )
        rows.append(
            {
                "city": city,
                "model": model,
                "N": len(frame),
                **result,
                "R2_reliable_N30": len(frame) >= 30,
            }
        )
    return pd.DataFrame(rows)


def add_rows(
    rows: list[dict[str, object]],
    sample_id: np.ndarray,
    city: np.ndarray,
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    outer_groups: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    outer_fold: int,
    model: str,
    prediction: np.ndarray,
    kd_weight: float,
) -> None:
    for position, index in enumerate(indices):
        rows.append(
            {
                "sample_id": sample_id[index],
                "city": city[index],
                "tile_x": tile_x[index],
                "tile_y": tile_y[index],
                "outer_group": outer_groups[index],
                "outer_fold": outer_fold,
                "model": model,
                "y_true": target[index],
                "y_pred": prediction[position],
                "kd_weight": kd_weight,
            }
        )


def main() -> None:
    args = parse_args()
    if args.shuffle_seed is None:
        args.shuffle_seed = args.spatial_seed
    device = resolve_device(args.device)
    kd_weights = parse_number_list(args.kd_weight_grid, float)
    if 0.0 not in kd_weights or any(weight < 0 for weight in kd_weights):
        raise ValueError("--kd-weight-grid must include 0 and no negatives.")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2.")

    archive = np.load(args.features, allow_pickle=False)
    sample_id = archive["sample_id"].astype(str)
    city = archive["city"].astype(str)
    tile_x = archive["tile_x"].astype(np.int64)
    tile_y = archive["tile_y"].astype(np.int64)
    target = transform_target(
        archive["target"].astype(np.float64), args.target_transform
    )
    satellite_raw = archive["satellite_features"].astype(np.float32)
    street_raw = archive["street_features"].astype(np.float32)
    reference, fold_assignment, outer_groups = load_reference(
        args.reference_dir, sample_id, city, target, args.spatial_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    isolation_rows: list[dict[str, object]] = []

    outer_progress = tqdm(
        sorted(np.unique(fold_assignment)),
        desc="CityLens prediction-KD controls",
        disable=args.no_progress,
    )
    for outer_fold in outer_progress:
        outer_test = np.flatnonzero(fold_assignment == outer_fold)
        outer_train = np.flatnonzero(fold_assignment != outer_fold)
        model_seed = args.seed + int(outer_fold)
        split_seed = args.spatial_seed + int(outer_fold)
        shuffle_seed = args.shuffle_seed + int(outer_fold)
        sat_scaler = StandardScaler().fit(satellite_raw[outer_train])
        street_scaler = StandardScaler().fit(street_raw[outer_train])
        satellite_scaled = sat_scaler.transform(satellite_raw).astype(np.float32)
        street_scaled = street_scaler.transform(street_raw).astype(np.float32)
        satellite = tensor(satellite_scaled, device)
        street = tensor(street_scaled, device)

        final_train, final_validation = spatial_train_validation_split(
            outer_train,
            outer_groups,
            args.validation_fraction,
            split_seed + 99,
        )
        crossfit_pool = final_train if args.strict_label_isolation else outer_train

        soft_predictions = {
            kind: np.full(len(target), np.nan, dtype=np.float64)
            for kind in ("paired", "shuffled", "satellite")
        }
        # Embeddings are ignored because relation_weight=0, but the existing
        # prediction-KD trainer expects a finite tensor with hidden_dim columns.
        dummy_embedding = np.zeros((len(target), args.hidden_dim), dtype=np.float32)
        inner_groups = outer_groups[crossfit_pool]
        inner_folds = min(args.inner_folds, len(np.unique(inner_groups)))
        inner_splitter = GroupKFold(n_splits=inner_folds)
        inner_progress = tqdm(
            enumerate(
                inner_splitter.split(
                    crossfit_pool, target[crossfit_pool], inner_groups
                ),
                start=1,
            ),
            total=inner_folds,
            desc=f"cross-fit teachers fold={outer_fold}",
            leave=False,
            disable=args.no_progress,
        )
        for inner_fold, (fit_position, hold_position) in inner_progress:
            inner_pool = crossfit_pool[fit_position]
            inner_hold = crossfit_pool[hold_position]
            inner_train, inner_validation = spatial_train_validation_split(
                inner_pool,
                outer_groups,
                args.validation_fraction,
                split_seed + inner_fold,
            )
            paired_teacher = train_teacher(
                satellite,
                street,
                target,
                inner_train,
                inner_validation,
                args,
                device,
                model_seed + 10 * inner_fold,
            )
            soft_predictions["paired"][inner_hold] = predict_teacher(
                paired_teacher, satellite, street, inner_hold
            )[0]

            shuffled_scaled, _ = shuffled_within_city(
                street_scaled,
                city,
                [inner_train, inner_validation, inner_hold],
                shuffle_seed + 1000 + inner_fold,
            )
            shuffled_street = tensor(shuffled_scaled, device)
            shuffled_teacher = train_teacher(
                satellite,
                shuffled_street,
                target,
                inner_train,
                inner_validation,
                args,
                device,
                model_seed + 10 * inner_fold,
            )
            soft_predictions["shuffled"][inner_hold] = predict_teacher(
                shuffled_teacher, satellite, shuffled_street, inner_hold
            )[0]

            satellite_teacher = train_supervised_student(
                satellite,
                target,
                inner_train,
                inner_validation,
                args,
                device,
                model_seed + 10 * inner_fold,
            )
            soft_predictions["satellite"][inner_hold] = predict_student(
                satellite_teacher, satellite, inner_hold
            )[0]

        for kind, values in soft_predictions.items():
            if not np.isfinite(values[crossfit_pool]).all():
                raise RuntimeError(f"Incomplete {kind} cross-fitted soft targets.")
        models = {}
        selected_weights = {}
        for kind in ("paired", "shuffled", "satellite"):
            if args.strict_label_isolation:
                (
                    fitted,
                    selected_weight,
                    validation_r2,
                    candidate_train,
                    candidate_earlystop,
                ) = select_kd_student_with_isolated_validation(
                    satellite=satellite,
                    target=target,
                    teacher_prediction=soft_predictions[kind],
                    teacher_embedding=dummy_embedding,
                    selection_train_indices=final_train,
                    selection_validation_indices=final_validation,
                    groups=outer_groups,
                    candidate_weights=kd_weights,
                    relation_weight=0.0,
                    validation_fraction=args.validation_fraction,
                    split_seed=split_seed + 700,
                    args=args,
                    device=device,
                    seed=model_seed + 100,
                )
            else:
                fitted, selected_weight, validation_r2 = (
                    select_kd_student_by_spatial_validation(
                        satellite=satellite,
                        target=target,
                        teacher_prediction=soft_predictions[kind],
                        teacher_embedding=dummy_embedding,
                        train_indices=final_train,
                        validation_indices=final_validation,
                        candidate_weights=kd_weights,
                        relation_weight=0.0,
                        args=args,
                        device=device,
                        seed=model_seed + 100,
                    )
                )
                candidate_train = final_train
                candidate_earlystop = final_validation
            models[kind] = fitted
            selected_weights[kind] = selected_weight
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "teacher_kind": kind,
                    "selected_weight": selected_weight,
                    "selected_validation_R2": validation_r2,
                    "strict_label_isolation": args.strict_label_isolation,
                }
            )

        if args.strict_label_isolation:
            isolation_rows.append(
                {
                    "outer_fold": outer_fold,
                    "N_outer_train": len(outer_train),
                    "N_selection_train": len(final_train),
                    "N_selection_validation": len(final_validation),
                    "N_candidate_train": len(candidate_train),
                    "N_candidate_earlystop": len(candidate_earlystop),
                    "selection_validation_in_teacher_crossfit_pool": len(
                        set(final_validation.tolist()).intersection(
                            crossfit_pool.tolist()
                        )
                    ),
                    "selection_validation_in_candidate_fit_or_earlystop": len(
                        set(final_validation.tolist()).intersection(
                            np.concatenate(
                                [candidate_train, candidate_earlystop]
                            ).tolist()
                        )
                    ),
                    "outer_test_in_any_internal_set": len(
                        set(outer_test.tolist()).intersection(
                            np.concatenate(
                                [crossfit_pool, final_validation]
                            ).tolist()
                        )
                    ),
                }
            )

        # All selectors train the exact same weight=0 candidate. Reuse paired's
        # selected model only when it selected zero; otherwise train a dedicated
        # no-privilege model with the identical student seed.
        if args.strict_label_isolation:
            no_privilege = select_kd_student_with_isolated_validation(
                satellite=satellite,
                target=target,
                teacher_prediction=soft_predictions["paired"],
                teacher_embedding=dummy_embedding,
                selection_train_indices=final_train,
                selection_validation_indices=final_validation,
                groups=outer_groups,
                candidate_weights=[0.0],
                relation_weight=0.0,
                validation_fraction=args.validation_fraction,
                split_seed=split_seed + 700,
                args=args,
                device=device,
                seed=model_seed + 100,
            )[0]
        else:
            no_privilege = select_kd_student_by_spatial_validation(
                satellite=satellite,
                target=target,
                teacher_prediction=soft_predictions["paired"],
                teacher_embedding=dummy_embedding,
                train_indices=final_train,
                validation_indices=final_validation,
                candidate_weights=[0.0],
                relation_weight=0.0,
                args=args,
                device=device,
                seed=model_seed + 100,
            )[0]
        predictions = {
            "prediction_kd_no_privilege": predict_student(
                no_privilege, satellite, outer_test
            )[0],
            "prediction_kd_paired": predict_student(
                models["paired"], satellite, outer_test
            )[0],
            "prediction_kd_shuffled": predict_student(
                models["shuffled"], satellite, outer_test
            )[0],
            "prediction_kd_satellite_teacher": predict_student(
                models["satellite"], satellite, outer_test
            )[0],
        }
        for kind, values in soft_predictions.items():
            signal_rows.append(
                {
                    "outer_fold": outer_fold,
                    "teacher_kind": kind,
                    "OOF_R2_on_outer_train": float(
                        r2_score(target[crossfit_pool], values[crossfit_pool])
                    ),
                    "OOF_MAE_on_outer_train": float(
                        mean_absolute_error(
                            target[crossfit_pool], values[crossfit_pool]
                        )
                    ),
                }
            )
        for model, prediction in predictions.items():
            weight = (
                0.0
                if model == "prediction_kd_no_privilege"
                else selected_weights[model.removeprefix("prediction_kd_").replace("_teacher", "")]
            )
            add_rows(
                prediction_rows,
                sample_id,
                city,
                tile_x,
                tile_y,
                outer_groups,
                target,
                outer_test,
                int(outer_fold),
                model,
                prediction,
                weight,
            )

    new_predictions = pd.DataFrame(prediction_rows)
    reference = reference.copy()
    reference["sample_id"] = reference["sample_id"].astype(str)
    reference["kd_weight"] = np.nan
    combined = pd.concat([reference, new_predictions], ignore_index=True, sort=False)
    combined.to_csv(args.output_dir / "outer_oof_predictions.csv", index=False)
    pooled = summarize(combined)
    pooled.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    fold_frames = []
    for fold, frame in combined.groupby("outer_fold"):
        summary = summarize(frame)
        summary.insert(0, "outer_fold", fold)
        fold_frames.append(summary)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        args.output_dir / "fold_metrics.csv", index=False
    )
    per_city = city_summary(combined)
    per_city.to_csv(args.output_dir / "city_metrics.csv", index=False)
    (
        per_city[per_city["R2_reliable_N30"]]
        .groupby("model", as_index=False)
        .agg(
            cities=("city", "nunique"),
            R2_macro=("R2", "mean"),
            RMSE_macro=("RMSE", "mean"),
            MAE_macro=("MAE", "mean"),
        )
        .to_csv(args.output_dir / "city_macro_summary.csv", index=False)
    )
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "kd_weight_selection.csv", index=False
    )
    pd.DataFrame(signal_rows).to_csv(
        args.output_dir / "teacher_signal_diagnostics.csv", index=False
    )
    if args.strict_label_isolation:
        pd.DataFrame(isolation_rows).to_csv(
            args.output_dir / "label_isolation_audit.csv", index=False
        )
    config = vars(args).copy()
    for key in ("features", "reference_dir", "output_dir"):
        config[key] = str(config[key])
    config["device_resolved"] = str(device)
    config["controls"] = ["paired", "shuffled", "satellite"]
    config["selection_protocol"] = (
        "nested_label_isolated" if args.strict_label_isolation else "less_isolated"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Pooled prediction-KD control summary")
    print(pooled.to_string(index=False))
    print("\nKD weight selection")
    print(pd.DataFrame(selection_rows).to_string(index=False))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
