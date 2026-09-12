"""Audit WorldPop 100 m source-cell overlap across Haidian outer folds.

This is a read-only audit.  It reproduces the KMeans spatial grouping and
GroupKFold outer assignment used by ``run_haidian_nested_spatial_controls.py``
and maps every sample to the actual WorldPop raster row/column.  The preferred
input is the pair of extracted feature archives, because those archives are
already restricted to samples with valid satellite/street-view imagery.  CSV
metadata mode is provided for a preliminary audit before feature extraction.

The main leakage check is whether one WorldPop 100 m cell occurs in more than
one outer fold.  Adjacent cross-fold cells and nearest cross-fold distances are
also reported because distinct raster cells can still be spatially dependent.
No model is trained and no input file is modified.

Feature-archive example:

    python audit_worldpop100m_crossfold.py \
        --features haidian_population100m_train_dinov2b14.npz \
                   haidian_population100m_test_dinov2b14.npz \
        --raster ./chn_pop_2020_CN_100m_R2025A_v1.tif \
        --output-dir worldpop100m_crossfold_audit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features",
        type=Path,
        nargs=2,
        metavar=("TRAIN_NPZ", "TEST_NPZ"),
        help="Two extracted feature archives: train and test.",
    )
    source.add_argument(
        "--metadata",
        type=Path,
        nargs=2,
        metavar=("TRAIN_CSV", "TEST_CSV"),
        help="Two relabeled CSVs for a preliminary audit before extraction.",
    )
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--spatial-blocks", type=int, default=25)
    parser.add_argument("--spatial-seed", type=int, default=42)
    return parser.parse_args()


def load_sources(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    if args.features is not None:
        frames: list[pd.DataFrame] = []
        for split, path in zip(("original_train", "original_test"), args.features, strict=True):
            archive = np.load(path, allow_pickle=False)
            required = {"sample_id", "latitude", "longitude"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {missing}")
            frame = pd.DataFrame(
                {
                    "sample_id": archive["sample_id"].astype(str),
                    "latitude": archive["latitude"].astype(float),
                    "longitude": archive["longitude"].astype(float),
                    "source_split": split,
                }
            )
            frames.append(frame)
        samples = pd.concat(frames, ignore_index=True)
        exact_feature_sample_set = True
    else:
        frames = []
        for split, path in zip(("original_train", "original_test"), args.metadata, strict=True):
            frame = pd.read_csv(path)
            required = {"Filename", "wgs84_lat", "wgs84_lng"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            frames.append(
                pd.DataFrame(
                    {
                        "sample_id": frame["Filename"].astype(str),
                        "latitude": pd.to_numeric(frame["wgs84_lat"], errors="coerce"),
                        "longitude": pd.to_numeric(frame["wgs84_lng"], errors="coerce"),
                        "source_split": split,
                    }
                )
            )
        samples = pd.concat(frames, ignore_index=True)
        exact_feature_sample_set = False

    if samples["sample_id"].duplicated().any():
        duplicate_ids = samples.loc[samples["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"Duplicate sample IDs across inputs: {duplicate_ids}")
    if not np.isfinite(samples[["latitude", "longitude"]].to_numpy(float)).all():
        raise ValueError("Inputs contain non-finite coordinates.")
    return samples, exact_feature_sample_set


def spatial_groups(latitude: np.ndarray, longitude: np.ndarray, n_blocks: int, seed: int) -> np.ndarray:
    mean_latitude_radians = np.deg2rad(np.mean(latitude))
    coordinates = np.column_stack([longitude * np.cos(mean_latitude_radians), latitude])
    coordinates = StandardScaler().fit_transform(coordinates)
    n_clusters = min(n_blocks, len(coordinates))
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(coordinates)


def assign_outer_folds(samples: pd.DataFrame, outer_folds: int, blocks: int, seed: int) -> pd.DataFrame:
    latitude = samples["latitude"].to_numpy(float)
    longitude = samples["longitude"].to_numpy(float)
    groups = spatial_groups(latitude, longitude, blocks, seed)
    if len(np.unique(groups)) < outer_folds:
        raise ValueError("Too few spatial groups for the requested outer folds.")
    fold_assignment = np.zeros(len(samples), dtype=np.int64)
    indices = np.arange(len(samples), dtype=np.int64)
    splitter = GroupKFold(n_splits=outer_folds)
    for fold, (_, test_position) in enumerate(splitter.split(indices, groups=groups), start=1):
        fold_assignment[test_position] = fold
    result = samples.copy()
    result["spatial_group"] = groups
    result["outer_fold"] = fold_assignment
    return result


def attach_raster_cells(samples: pd.DataFrame, raster_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    with rasterio.open(raster_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Raster has no CRS.")
        longitude = samples["longitude"].to_numpy(float)
        latitude = samples["latitude"].to_numpy(float)
        if dataset.crs.to_string() == "EPSG:4326":
            xs, ys = longitude.tolist(), latitude.tolist()
        else:
            xs, ys = transform_coordinates("EPSG:4326", dataset.crs, longitude.tolist(), latitude.tolist())
        points = list(zip(xs, ys, strict=True))
        rows, cols = zip(*(dataset.index(x, y) for x, y in points), strict=True)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = list(dataset.sample(points, indexes=1, masked=True))
        masked = np.asarray([np.ma.is_masked(item[0]) for item in values], dtype=bool)
        raw = np.asarray(
            [float(item[0]) if not np.ma.is_masked(item[0]) else np.nan for item in values],
            dtype=float,
        )
        outside = (rows < 0) | (rows >= dataset.height) | (cols < 0) | (cols >= dataset.width)
        invalid = outside | masked | ~np.isfinite(raw) | (raw < 0)
        result = samples.copy()
        result["worldpop100m_row"] = rows
        result["worldpop100m_col"] = cols
        result["worldpop100m_value"] = raw
        result["worldpop100m_invalid"] = invalid
        result["worldpop100m_cell"] = [f"{r}:{c}" for r, c in zip(rows, cols, strict=True)]
        raster_info = {
            "path": str(raster_path.resolve()),
            "crs": dataset.crs.to_string(),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "transform": [float(value) for value in dataset.transform],
            "resolution": [float(value) for value in dataset.res],
            "bounds": {
                "left": float(dataset.bounds.left),
                "bottom": float(dataset.bounds.bottom),
                "right": float(dataset.bounds.right),
                "top": float(dataset.bounds.top),
            },
            "nodata": dataset.nodata,
            "invalid_samples": int(invalid.sum()),
            "outside_index_bounds": int(outside.sum()),
            "masked_samples": int(masked.sum()),
        }
    return result, raster_info


def local_xy(samples: pd.DataFrame) -> np.ndarray:
    mean_latitude = math.radians(float(samples["latitude"].mean()))
    radius = 6_371_000.0
    return np.column_stack(
        [
            radius * np.deg2rad(samples["longitude"].to_numpy(float)) * math.cos(mean_latitude),
            radius * np.deg2rad(samples["latitude"].to_numpy(float)),
        ]
    )


def nearest_cross_fold_distance(samples: pd.DataFrame) -> np.ndarray:
    xy = local_xy(samples)
    folds = samples["outer_fold"].to_numpy(int)
    distances = np.full(len(samples), np.nan, dtype=float)
    for fold in np.unique(folds):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        distances[test] = cKDTree(xy[train]).query(xy[test], k=1)[0]
    return distances


def cross_fold_cell_summary(samples: pd.DataFrame) -> dict[str, int]:
    grouped = samples.groupby("worldpop100m_cell", sort=False)
    cross = [group for _, group in grouped if group["outer_fold"].nunique() > 1]
    return {
        "occupied_cells": int(len(grouped)),
        "cells_spanning_multiple_outer_folds": int(len(cross)),
        "rows_in_cells_spanning_multiple_outer_folds": int(sum(len(group) for group in cross)),
    }


def adjacent_cell_summary(samples: pd.DataFrame) -> dict[str, int]:
    cells_by_fold: dict[tuple[int, int], set[int]] = {}
    for row in samples[["worldpop100m_row", "worldpop100m_col", "outer_fold"]].itertuples(index=False):
        cells_by_fold.setdefault((int(row.worldpop100m_row), int(row.worldpop100m_col)), set()).add(int(row.outer_fold))
    occupied = set(cells_by_fold)
    adjacent_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    adjacent_cells: set[tuple[int, int]] = set()
    for row, col in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (row + dr, col + dc)
                if neighbor not in occupied:
                    continue
                if cells_by_fold[(row, col)] == cells_by_fold[neighbor]:
                    continue
                adjacent_cells.update(((row, col), neighbor))
                adjacent_pairs.add(tuple(sorted(((row, col), neighbor))))
    return {
        "occupied_cells_with_cross_fold_neighbors": int(len(adjacent_cells)),
        "cross_fold_adjacent_cell_pairs": int(len(adjacent_pairs)),
    }


def main() -> None:
    args = parse_args()
    raster_path = args.raster.expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    samples, exact_feature_sample_set = load_sources(args)
    samples = assign_outer_folds(samples, args.outer_folds, args.spatial_blocks, args.spatial_seed)
    samples, raster_info = attach_raster_cells(samples, raster_path)
    distances = nearest_cross_fold_distance(samples)
    samples["nearest_cross_fold_distance_m"] = distances

    invalid = samples["worldpop100m_invalid"].to_numpy(bool)
    valid = samples.loc[~invalid].copy()
    summary: dict[str, object] = {
        "sample_count": int(len(samples)),
        "exact_feature_sample_set": bool(exact_feature_sample_set),
        "outer_folds": int(args.outer_folds),
        "spatial_blocks": int(args.spatial_blocks),
        "spatial_seed": int(args.spatial_seed),
        "raster": raster_info,
        "cross_fold_cell_check": cross_fold_cell_summary(valid) if len(valid) else {},
        "adjacent_cross_fold_cell_check": adjacent_cell_summary(valid) if len(valid) else {},
        "nearest_cross_fold_distance_m": {
            "min": float(np.nanmin(distances)),
            "p05": float(np.nanpercentile(distances, 5)),
            "median": float(np.nanmedian(distances)),
            "p95": float(np.nanpercentile(distances, 95)),
            "max": float(np.nanmax(distances)),
        },
        "fold_counts": {str(int(fold)): int(count) for fold, count in samples["outer_fold"].value_counts().sort_index().items()},
    }
    fold_rows: list[dict[str, object]] = []
    for fold, frame in samples.groupby("outer_fold", sort=True):
        fold_rows.append(
            {
                "outer_fold": int(fold),
                "samples": int(len(frame)),
                "valid_worldpop_samples": int((~frame["worldpop100m_invalid"]).sum()),
                "invalid_worldpop_samples": int(frame["worldpop100m_invalid"].sum()),
                "spatial_groups": int(frame["spatial_group"].nunique()),
                "occupied_worldpop100m_cells": int(frame.loc[~frame["worldpop100m_invalid"], "worldpop100m_cell"].nunique()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_dir / "worldpop100m_crossfold_assignments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "worldpop100m_crossfold_folds.csv", index=False)
    (args.output_dir / "worldpop100m_crossfold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["cross_fold_cell_check"].get("cells_spanning_multiple_outer_folds", 0):
        print("WARNING: at least one WorldPop 100 m cell spans multiple outer folds.")
    else:
        print("PASS: no valid WorldPop 100 m cell spans multiple outer folds.")


if __name__ == "__main__":
    main()
