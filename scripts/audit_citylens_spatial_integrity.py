"""Audit CityLens cross-fold image reuse and nearest tile separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--satellite-root", type=Path, required=True)
    parser.add_argument("--street-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="sat_supervised")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse-hashes", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    return f"{int(''.join('1' if value else '0' for value in bits.ravel()), 2):016x}"


def collisions(frame: pd.DataFrame, column: str) -> tuple[int, int]:
    groups = frame.groupby(column)
    selected = groups.filter(lambda group: group["outer_fold"].nunique() > 1)
    return int(sum(g["outer_fold"].nunique() > 1 for _, g in groups)), int(len(selected))


def nearest_cross_fold_tiles(frame: pd.DataFrame) -> np.ndarray:
    output = np.full(len(frame), np.nan)
    for city in frame["city"].unique():
        city_idx = np.flatnonzero(frame["city"].to_numpy() == city)
        xy = frame.loc[city_idx, ["tile_x", "tile_y"]].to_numpy(float)
        folds = frame.loc[city_idx, "outer_fold"].to_numpy()
        for fold in np.unique(folds):
            test_local = np.flatnonzero(folds == fold)
            train_local = np.flatnonzero(folds != fold)
            if not len(test_local) or not len(train_local):
                continue
            output[city_idx[test_local]] = cKDTree(xy[train_local]).query(xy[test_local], k=1)[0]
    return output


def hash_job(job: tuple[str, int, str, Path]) -> dict[str, object]:
    sample_id, outer_fold, modality, path = job
    exists = path.is_file()
    return {
        "sample_id": sample_id,
        "outer_fold": outer_fold,
        "modality": modality,
        "path": str(path),
        "exists": exists,
        "sha256": sha256(path) if exists else "",
        "dhash": dhash(path) if exists else "",
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.oof)
    frame = frame[frame["model"] == args.model].copy()
    if frame["sample_id"].duplicated().any():
        raise ValueError("Selected model must provide one row per sample.")
    frame = frame.reset_index(drop=True)
    distance = nearest_cross_fold_tiles(frame)

    jobs: list[tuple[str, int, str, Path]] = []
    for sample in frame.itertuples(index=False):
        satellite = args.satellite_root / sample.city / f"{sample.sample_id}.png"
        paths = [("satellite", satellite)]
        street_dir = args.street_root / str(sample.sample_id)
        paths.extend(
            ("street", path)
            for path in sorted(street_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for modality, path in paths:
            jobs.append((str(sample.sample_id), int(sample.outer_fold), modality, path))
    reused: dict[str, tuple[str, str]] = {}
    if args.reuse_hashes and args.reuse_hashes.is_file():
        previous = pd.read_csv(args.reuse_hashes)
        previous = previous[previous["exists"] == True]  # noqa: E712
        reused = {
            str(row.path): (str(row.sha256), str(row.dhash))
            for row in previous.itertuples(index=False)
        }
    rows = []
    pending = []
    for sample_id, outer_fold, modality, path in jobs:
        if str(path) in reused:
            digest, perceptual = reused[str(path)]
            rows.append(
                {
                    "sample_id": sample_id,
                    "outer_fold": outer_fold,
                    "modality": modality,
                    "path": str(path),
                    "exists": True,
                    "sha256": digest,
                    "dhash": perceptual,
                }
            )
        else:
            pending.append((sample_id, outer_fold, modality, path))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows.extend(
            tqdm(executor.map(hash_job, pending), total=len(pending), desc="image hashes")
        )
    hashes = pd.DataFrame(rows)
    valid = hashes[hashes["exists"]]
    summary: dict[str, object] = {
        "samples": int(len(frame)),
        "cities": int(frame["city"].nunique()),
        "outer_folds": int(frame["outer_fold"].nunique()),
        "missing_image_files": int((~hashes["exists"]).sum()),
        "nearest_cross_fold_distance_tiles_min": float(np.nanmin(distance)),
        "nearest_cross_fold_distance_tiles_p05": float(np.nanpercentile(distance, 5)),
        "nearest_cross_fold_distance_tiles_median": float(np.nanmedian(distance)),
        "nearest_cross_fold_distance_tiles_p95": float(np.nanpercentile(distance, 95)),
        "nearest_cross_fold_distance_tiles_max": float(np.nanmax(distance)),
    }
    for modality in ("satellite", "street"):
        part = valid[valid["modality"] == modality]
        for field in ("sha256", "dhash"):
            group_count, row_count = collisions(part, field)
            summary[f"{modality}_{field}_cross_fold_groups"] = group_count
            summary[f"{modality}_{field}_rows_in_cross_fold_groups"] = row_count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.assign(nearest_cross_fold_distance_tiles=distance).to_csv(
        args.output_dir / "sample_integrity_diagnostics.csv", index=False
    )
    hashes.to_csv(args.output_dir / "image_hash_diagnostics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
