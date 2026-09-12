"""Shared training and evaluation utilities for the street-view audit.

Teachers see satellite and street-view features during fitting; every student
evaluated at inference receives satellite features only. Outer folds are
untouched spatial test folds, and the KD-weight selector keeps selection
validation labels out of teacher cross-fitting and candidate early stopping.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from tqdm.auto import tqdm

from models import (
    MultimodalTeacher,
    SatelliteStudent,
    relational_distillation_loss,
    seed_everything,
)


@dataclass
class FittedRegressor:
    model: torch.nn.Module
    target_mean: float
    target_std: float


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def iterate_batches(
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[np.ndarray]:
    order = np.asarray(indices).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def spatial_groups(
    latitude: np.ndarray,
    longitude: np.ndarray,
    n_blocks: int,
    seed: int,
) -> np.ndarray:
    """Create compact spatial groups using locally projected coordinates."""
    mean_latitude_radians = np.deg2rad(np.mean(latitude))
    coordinates = np.column_stack(
        [longitude * np.cos(mean_latitude_radians), latitude]
    )
    coordinates = StandardScaler().fit_transform(coordinates)
    n_clusters = min(n_blocks, len(coordinates))
    return KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    ).fit_predict(coordinates)


def spatial_train_validation_split(
    indices: np.ndarray,
    groups: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train_position, validation_position = next(
        splitter.split(indices, groups=groups[indices])
    )
    return indices[train_position], indices[validation_position]


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def deranged_modality_copy(
    modality: np.ndarray,
    partitions: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Shuffle rows within each split without retaining any original pairing."""
    shuffled = modality.copy()
    rng = np.random.default_rng(seed)
    for indices in partitions:
        indices = np.asarray(indices)
        if len(indices) < 2:
            raise ValueError("A shuffled-modality partition must contain >=2 samples.")
        destination_order = rng.permutation(indices)
        source_order = np.roll(destination_order, 1)
        shuffled[destination_order] = modality[source_order]
    return shuffled


def train_supervised_student(
    satellite: torch.Tensor,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> FittedRegressor:
    seed_everything(seed)
    model = SatelliteStudent(
        satellite_dim=satellite.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    return _fit_supervised(
        model=model,
        satellite=satellite,
        street=None,
        target=target,
        train_indices=train_indices,
        validation_indices=validation_indices,
        args=args,
        teacher=False,
        seed=seed,
    )


def train_teacher(
    satellite: torch.Tensor,
    street: torch.Tensor,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> FittedRegressor:
    seed_everything(seed)
    model = MultimodalTeacher(
        satellite_dim=satellite.shape[1],
        street_dim=street.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    return _fit_supervised(
        model=model,
        satellite=satellite,
        street=street,
        target=target,
        train_indices=train_indices,
        validation_indices=validation_indices,
        args=args,
        teacher=True,
        seed=seed,
    )


def _fit_supervised(
    model: torch.nn.Module,
    satellite: torch.Tensor,
    street: torch.Tensor | None,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    teacher: bool,
    seed: int,
) -> FittedRegressor:
    target_mean = float(np.mean(target[train_indices]))
    target_std = float(np.std(target[train_indices]))
    target_std = max(target_std, 1e-6)
    target_scaled = tensor((target - target_mean) / target_std, satellite.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0

    epoch_progress = tqdm(
        range(args.epochs),
        desc=f"fit {model.__class__.__name__}",
        leave=False,
        disable=args.no_progress,
    )
    for epoch in epoch_progress:
        model.train()
        for batch_indices in iterate_batches(
            train_indices, args.batch_size, True, seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            if teacher:
                prediction, _, _ = model(
                    satellite[batch_indices], street[batch_indices]  # type: ignore[index]
                )
            else:
                prediction, _ = model(satellite[batch_indices])
            loss = F.smooth_l1_loss(prediction, target_scaled[batch_indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if teacher:
                validation_prediction, _, _ = model(
                    satellite[validation_indices],
                    street[validation_indices],  # type: ignore[index]
                )
            else:
                validation_prediction, _ = model(satellite[validation_indices])
            validation_loss = F.smooth_l1_loss(
                validation_prediction,
                target_scaled[validation_indices],
            ).item()

        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        epoch_progress.set_postfix(
            val=f"{validation_loss:.4f}",
            stale=f"{stale_epochs}/{args.patience}",
        )

    model.load_state_dict(best_state)
    return FittedRegressor(model, target_mean, target_std)


@torch.no_grad()
def predict_student(
    fitted: FittedRegressor,
    satellite: torch.Tensor,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fitted.model.eval()
    prediction, embedding = fitted.model(satellite[indices])
    prediction = prediction.cpu().numpy() * fitted.target_std + fitted.target_mean
    return prediction, embedding.cpu().numpy()


@torch.no_grad()
def predict_teacher(
    fitted: FittedRegressor,
    satellite: torch.Tensor,
    street: torch.Tensor,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitted.model.eval()
    prediction, embedding, gate = fitted.model(
        satellite[indices], street[indices]
    )
    prediction = prediction.cpu().numpy() * fitted.target_std + fitted.target_mean
    return prediction, embedding.cpu().numpy(), gate.cpu().numpy()


def train_kd_student(
    satellite: torch.Tensor,
    target: np.ndarray,
    teacher_prediction: np.ndarray,
    teacher_embedding: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    kd_weight: float,
    relation_weight: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> FittedRegressor:
    seed_everything(seed)
    model = SatelliteStudent(
        satellite_dim=satellite.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    target_mean = float(np.mean(target[train_indices]))
    target_std = max(float(np.std(target[train_indices])), 1e-6)
    target_scaled = tensor((target - target_mean) / target_std, device)
    teacher_prediction_scaled = tensor(
        (teacher_prediction - target_mean) / target_std,
        device,
    )
    teacher_embedding_tensor = tensor(teacher_embedding, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0

    kd_label = "relation KD" if relation_weight > 0 else "prediction KD"
    epoch_progress = tqdm(
        range(args.epochs),
        desc=f"fit {kd_label} (w={kd_weight:g})",
        leave=False,
        disable=args.no_progress,
    )
    for epoch in epoch_progress:
        model.train()
        for batch_indices in iterate_batches(
            train_indices, args.batch_size, True, seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction, embedding = model(satellite[batch_indices])
            supervised_loss = F.smooth_l1_loss(
                prediction, target_scaled[batch_indices]
            )
            prediction_kd_loss = F.smooth_l1_loss(
                prediction, teacher_prediction_scaled[batch_indices]
            )
            relation_loss = relational_distillation_loss(
                embedding,
                teacher_embedding_tensor[batch_indices],
            )
            loss = (
                supervised_loss
                + kd_weight * args.lambda_pred * prediction_kd_loss
                + kd_weight * relation_weight * relation_loss
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(satellite[validation_indices])
            validation_loss = F.smooth_l1_loss(
                validation_prediction,
                target_scaled[validation_indices],
            ).item()
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        epoch_progress.set_postfix(
            val=f"{validation_loss:.4f}",
            stale=f"{stale_epochs}/{args.patience}",
        )

    model.load_state_dict(best_state)
    return FittedRegressor(model, target_mean, target_std)


def select_kd_student_by_spatial_validation(
    satellite: torch.Tensor,
    target: np.ndarray,
    teacher_prediction: np.ndarray,
    teacher_embedding: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    candidate_weights: list[float],
    relation_weight: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[FittedRegressor, float, float]:
    """Select KD strength by student gain on a spatial validation set.

    This is the less-isolated protocol: the same validation set used for
    candidate early stopping also ranks the candidates. All candidates start
    from the same initialization, and including weight 0 allows an exact
    fallback to supervised satellite training.
    """
    if 0.0 not in candidate_weights:
        raise ValueError("--kd-weight-grid must include 0 for a safe fallback.")
    best_model: FittedRegressor | None = None
    best_weight = 0.0
    best_validation_r2 = -float("inf")

    weight_progress = tqdm(
        sorted(set(candidate_weights)),
        desc="select KD weight",
        leave=False,
        disable=args.no_progress,
    )
    for candidate_weight in weight_progress:
        candidate_model = train_kd_student(
            satellite=satellite,
            target=target,
            teacher_prediction=teacher_prediction,
            teacher_embedding=teacher_embedding,
            train_indices=train_indices,
            validation_indices=validation_indices,
            kd_weight=candidate_weight,
            relation_weight=relation_weight,
            args=args,
            device=device,
            seed=seed,
        )
        validation_prediction, _ = predict_student(
            candidate_model,
            satellite,
            validation_indices,
        )
        validation_r2 = float(r2_score(target[validation_indices], validation_prediction))
        # Strict improvement preserves the smaller KD weight when effectively tied.
        if validation_r2 > best_validation_r2 + 1e-8:
            best_model = candidate_model
            best_weight = candidate_weight
            best_validation_r2 = validation_r2
        weight_progress.set_postfix(
            candidate=f"{candidate_weight:g}",
            best=f"{best_weight:g}",
            val_r2=f"{validation_r2:.4f}",
        )

    if best_model is None:
        raise RuntimeError("KD validation failed to produce a model.")
    return best_model, best_weight, best_validation_r2


def select_kd_student_with_isolated_validation(
    satellite: torch.Tensor,
    target: np.ndarray,
    teacher_prediction: np.ndarray,
    teacher_embedding: np.ndarray,
    selection_train_indices: np.ndarray,
    selection_validation_indices: np.ndarray,
    groups: np.ndarray,
    candidate_weights: list[float],
    relation_weight: float,
    validation_fraction: float,
    split_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[FittedRegressor, float, float, np.ndarray, np.ndarray]:
    """Select KD weight without using selection-validation labels in fitting.

    Candidate early stopping uses a spatially disjoint subset carved only from
    ``selection_train_indices``. The untouched selection-validation labels are
    read only to rank candidate weights. The winning candidate is returned
    without refitting, so those labels never affect model parameters, early
    stopping, teacher fitting, or soft-target construction.
    """
    if 0.0 not in candidate_weights:
        raise ValueError("KD candidate weights must include zero.")
    candidate_train, candidate_earlystop = spatial_train_validation_split(
        selection_train_indices,
        groups,
        validation_fraction,
        split_seed,
    )
    selection_validation_set = set(selection_validation_indices.tolist())
    if selection_validation_set.intersection(candidate_train.tolist()):
        raise RuntimeError("Selection-validation rows entered candidate training.")
    if selection_validation_set.intersection(candidate_earlystop.tolist()):
        raise RuntimeError("Selection-validation rows entered candidate early stopping.")
    if not np.isfinite(teacher_prediction[selection_train_indices]).all():
        raise RuntimeError("Teacher targets are incomplete on selection training rows.")

    best_weight = 0.0
    best_validation_r2 = -float("inf")
    best_model: FittedRegressor | None = None
    weight_progress = tqdm(
        sorted(set(candidate_weights)),
        desc="select KD weight (label-isolated)",
        leave=False,
        disable=args.no_progress,
    )
    for candidate_weight in weight_progress:
        candidate_model = train_kd_student(
            satellite=satellite,
            target=target,
            teacher_prediction=teacher_prediction,
            teacher_embedding=teacher_embedding,
            train_indices=candidate_train,
            validation_indices=candidate_earlystop,
            kd_weight=candidate_weight,
            relation_weight=relation_weight,
            args=args,
            device=device,
            seed=seed,
        )
        validation_prediction, _ = predict_student(
            candidate_model, satellite, selection_validation_indices
        )
        validation_r2 = float(
            r2_score(target[selection_validation_indices], validation_prediction)
        )
        if validation_r2 > best_validation_r2 + 1e-8:
            best_model = candidate_model
            best_weight = candidate_weight
            best_validation_r2 = validation_r2
        weight_progress.set_postfix(
            candidate=f"{candidate_weight:g}",
            best=f"{best_weight:g}",
            val_r2=f"{validation_r2:.4f}",
        )

    if best_model is None:
        raise RuntimeError("Label-isolated KD selection failed to produce a model.")
    return (
        best_model,
        best_weight,
        best_validation_r2,
        candidate_train,
        candidate_earlystop,
    )


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(target, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(target, prediction))),
        "MAE": float(mean_absolute_error(target, prediction)),
    }


