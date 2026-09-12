"""Attach WorldPop 100 m population-count labels to Haidian CSV files.

The WorldPop file used here is a population-count raster: each pixel stores
people per pixel, not population density per square kilometre.  This script
samples the pixel containing each WGS84 point, computes ``log1p(count)``, and
writes a new CSV without modifying the source CSV.

By default, the output keeps the original target as ``population_original``
and replaces ``population`` with the new ``log1p`` target.  The raw raster
value is retained as ``population_worldpop100m_people_per_pixel`` and the
transformed value as ``population_worldpop100m``.  This naming makes the
target semantics explicit while remaining directly compatible with
``extract_dinov2_features.py``.

The script is intentionally limited to label preparation.  It does not read
or modify imagery, feature archives, model checkpoints, or experiment
results.  A JSON audit file records raster metadata, sampling settings,
coordinate coverage, nodata handling, and output statistics.

Example:

    python relabel_haidian_worldpop_100m.py \
        --input-csv ./data/Haidian_train.csv ./data/Haidian_test.csv \
        --raster /data/chn_pop_2020/CN/100m/R2025A/v1.tif \
        --output-dir ./data/worldpop100m \
        --raster-checksum

The raster product is the WorldPop Population Counts 2020 release.  Record
the exact downloaded filename, checksum, and download date in the experiment
configuration before reporting results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.warp import transform as transform_coordinates
except ImportError as error:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "This script requires rasterio. Install it in the experiment environment "
        "with `python -m pip install rasterio` before running."
    ) from error


LATITUDE_COLUMN = "wgs84_lat"
LONGITUDE_COLUMN = "wgs84_lng"
TARGET_COLUMN = "population"
RAW_OUTPUT_COLUMN = "population_worldpop100m_people_per_pixel"
LOG_OUTPUT_COLUMN = "population_worldpop100m"
ORIGINAL_OUTPUT_COLUMN = "population_original"
ROW_OUTPUT_COLUMN = "worldpop100m_row"
COL_OUTPUT_COLUMN = "worldpop100m_col"
MASK_OUTPUT_COLUMN = "worldpop100m_is_nodata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Haidian CSV files containing WGS84 coordinates.",
    )
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for relabeled CSV files and audit JSON files.",
    )
    parser.add_argument(
        "--latitude-column",
        default=LATITUDE_COLUMN,
        help=f"Latitude column (default: {LATITUDE_COLUMN}).",
    )
    parser.add_argument(
        "--longitude-column",
        default=LONGITUDE_COLUMN,
        help=f"Longitude column (default: {LONGITUDE_COLUMN}).",
    )
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help=(
            "Column to replace in the output CSV (default: population). "
            "Use --no-replace-target to retain the source value."
        ),
    )
    parser.add_argument(
        "--no-replace-target",
        action="store_true",
        help="Do not replace the target column; still write the new label columns.",
    )
    parser.add_argument(
        "--allow-nodata",
        action="store_true",
        help=(
            "Write NaN for raster nodata/out-of-bounds points instead of failing. "
            "This is not recommended for the downstream feature extractor."
        ),
    )
    parser.add_argument(
        "--drop-nodata",
        action="store_true",
        help=(
            "Drop rows whose sampled pixel is nodata, negative, or non-finite. "
            "The dropped row positions are recorded in the audit JSON."
        ),
    )
    parser.add_argument(
        "--raster-checksum",
        action="store_true",
        help="Compute a SHA-256 checksum of the raster for provenance auditing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output CSV or audit files.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Convert numpy/raster metadata values to JSON-compatible objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def output_paths(input_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = input_csv.stem
    return (
        output_dir / f"{stem}_worldpop100m.csv",
        output_dir / f"{stem}_worldpop100m_audit.json",
    )


def check_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace it: "
            + ", ".join(existing)
        )


def sample_raster(
    dataset: rasterio.io.DatasetReader,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample band 1 with nearest-neighbour semantics and return audit fields."""
    if dataset.count < 1:
        raise ValueError("The raster has no bands.")
    if dataset.crs is None:
        raise ValueError("The raster has no CRS; refusing to guess coordinate semantics.")

    if dataset.crs.to_string() == "EPSG:4326":
        xs = longitudes.tolist()
        ys = latitudes.tolist()
    else:
        xs, ys = transform_coordinates(
            "EPSG:4326",
            dataset.crs,
            longitudes.tolist(),
            latitudes.tolist(),
        )

    points = list(zip(xs, ys, strict=True))
    sampled = list(dataset.sample(points, indexes=1, masked=True))
    values = np.asarray(
        [float(item[0]) if not np.ma.is_masked(item[0]) else np.nan for item in sampled],
        dtype=np.float64,
    )
    rows, cols = zip(
        *(dataset.index(x, y) for x, y in points), strict=True
    )
    rows_array = np.asarray(rows, dtype=np.int64)
    cols_array = np.asarray(cols, dtype=np.int64)
    is_nodata = ~np.isfinite(values) | (values < 0)
    return values, rows_array, cols_array, is_nodata


def validate_coordinates(
    dataframe: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        column
        for column in (latitude_column, longitude_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"CSV is missing coordinate columns: {missing}")
    latitudes = pd.to_numeric(dataframe[latitude_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    longitudes = pd.to_numeric(dataframe[longitude_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= -90.0)
        & (latitudes <= 90.0)
        & (longitudes >= -180.0)
        & (longitudes <= 180.0)
    )
    if not valid.all():
        bad_rows = np.flatnonzero(~valid)[:10].tolist()
        raise ValueError(
            f"Invalid WGS84 coordinates in {int((~valid).sum())} rows; "
            f"first row positions: {bad_rows}"
        )
    return latitudes, longitudes


def relabel_dataframe(
    dataframe: pd.DataFrame,
    dataset: rasterio.io.DatasetReader,
    latitude_column: str,
    longitude_column: str,
    target_column: str,
    replace_target: bool,
    allow_nodata: bool,
    drop_nodata: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if allow_nodata and drop_nodata:
        raise ValueError("Use either --allow-nodata or --drop-nodata, not both.")
    latitudes, longitudes = validate_coordinates(
        dataframe, latitude_column, longitude_column
    )
    raw_values, rows, cols, is_nodata = sample_raster(dataset, latitudes, longitudes)
    nodata_count = int(is_nodata.sum())
    if nodata_count and not allow_nodata and not drop_nodata:
        examples = np.flatnonzero(is_nodata)[:10]
        detail_lines = []
        for position in examples:
            detail_lines.append(
                "  position={position}, lat={lat:.8f}, lon={lon:.8f}, "
                "row={row}, col={col}, sampled={sampled!r}".format(
                    position=int(position),
                    lat=float(latitudes[position]),
                    lon=float(longitudes[position]),
                    row=int(rows[position]),
                    col=int(cols[position]),
                    sampled=(
                        None
                        if not np.isfinite(raw_values[position])
                        else float(raw_values[position])
                    ),
                )
            )
        outside_count = int(
            (
                (rows < 0)
                | (rows >= dataset.height)
                | (cols < 0)
                | (cols >= dataset.width)
            ).sum()
        )
        detail_text = "\n".join(detail_lines)
        raise ValueError(
            f"{nodata_count} points returned nodata, negative, or non-finite values; "
            f"{outside_count} are outside the raster index bounds.\n"
            f"Raster CRS={dataset.crs}, bounds={dataset.bounds}, "
            f"nodata={dataset.nodata!r}\n"
            f"First invalid samples:\n{detail_text}\n"
            "Inspect the cause before using --allow-nodata; invalid rows must "
            "be removed before training."
        )

    source_positions = np.arange(len(dataframe), dtype=np.int64)
    if drop_nodata:
        keep = ~is_nodata
        if not keep.any():
            raise ValueError("All rows are nodata or invalid; nothing can be written.")
        output = dataframe.loc[keep].copy().reset_index(drop=True)
        raw_values = raw_values[keep]
        rows = rows[keep]
        cols = cols[keep]
        is_nodata = is_nodata[keep]
        source_positions = source_positions[keep]
    else:
        output = dataframe.copy()

    transformed = np.full(raw_values.shape, np.nan, dtype=np.float64)
    valid = ~is_nodata
    transformed[valid] = np.log1p(raw_values[valid])
    if target_column in output.columns:
        output[ORIGINAL_OUTPUT_COLUMN] = output[target_column]
    elif replace_target:
        raise ValueError(
            f"Cannot replace missing target column {target_column!r}; use "
            "--no-replace-target or supply a CSV containing that column."
        )
    output[RAW_OUTPUT_COLUMN] = raw_values.astype(np.float32)
    output[LOG_OUTPUT_COLUMN] = transformed.astype(np.float32)
    output[ROW_OUTPUT_COLUMN] = rows
    output[COL_OUTPUT_COLUMN] = cols
    output[MASK_OUTPUT_COLUMN] = is_nodata
    if replace_target:
        output[target_column] = transformed.astype(np.float32)

    audit = {
        "input_rows": int(len(dataframe)),
        "output_rows": int(len(output)),
        "nodata_or_invalid_samples": nodata_count,
        "valid_samples": int(valid.sum()),
        "dropped_source_row_positions": (
            np.flatnonzero(~np.isin(np.arange(len(dataframe)), source_positions))
            .astype(int)
            .tolist()
            if drop_nodata
            else []
        ),
        "raw_people_per_pixel": {
            "min": float(np.nanmin(raw_values)) if valid.any() else None,
            "median": float(np.nanmedian(raw_values)) if valid.any() else None,
            "max": float(np.nanmax(raw_values)) if valid.any() else None,
        },
        "log1p_target": {
            "min": float(np.nanmin(transformed)) if valid.any() else None,
            "median": float(np.nanmedian(transformed)) if valid.any() else None,
            "max": float(np.nanmax(transformed)) if valid.any() else None,
        },
        "target_replaced": bool(replace_target),
        "target_column": target_column,
        "drop_nodata": bool(drop_nodata),
    }
    return output, audit


def raster_metadata(
    dataset: rasterio.io.DatasetReader,
    raster_path: Path,
    include_checksum: bool,
) -> dict[str, object]:
    bounds = dataset.bounds
    metadata: dict[str, object] = {
        "path": str(raster_path.resolve()),
        "file_size_bytes": int(raster_path.stat().st_size),
        "driver": dataset.driver,
        "width": int(dataset.width),
        "height": int(dataset.height),
        "count": int(dataset.count),
        "dtype": list(dataset.dtypes),
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": list(dataset.transform),
        "resolution": [float(value) for value in dataset.res],
        "bounds": {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        },
        "nodata": dataset.nodata,
        "descriptions": list(dataset.descriptions),
        "units": list(dataset.units),
        "tags": {str(key): str(value) for key, value in dataset.tags().items()},
    }
    if include_checksum:
        print(f"Computing SHA-256: {raster_path}")
        metadata["sha256"] = sha256_file(raster_path)
    return metadata


def main() -> None:
    args = parse_args()
    raster_path = args.raster.expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_pairs = [output_paths(path, args.output_dir) for path in args.input_csv]
    check_output_paths(
        [path for pair in output_pairs for path in pair], args.overwrite
    )

    with rasterio.open(raster_path) as dataset:
        raster_info = raster_metadata(dataset, raster_path, args.raster_checksum)
        if dataset.count != 1:
            print(
                f"Warning: raster has {dataset.count} bands; sampling band 1 only."
            )
        if dataset.nodata is None:
            print(
                "Warning: raster does not declare a nodata value; non-finite and "
                "negative samples will still be rejected."
            )

        for input_csv, (output_csv, audit_json) in zip(
            args.input_csv, output_pairs, strict=True
        ):
            input_csv = input_csv.expanduser().resolve()
            if not input_csv.is_file():
                raise FileNotFoundError(f"Input CSV not found: {input_csv}")
            dataframe = pd.read_csv(input_csv)
            relabeled, label_audit = relabel_dataframe(
                dataframe=dataframe,
                dataset=dataset,
                latitude_column=args.latitude_column,
                longitude_column=args.longitude_column,
                target_column=args.target_column,
                replace_target=not args.no_replace_target,
                allow_nodata=args.allow_nodata,
                drop_nodata=args.drop_nodata,
            )
            relabeled.to_csv(output_csv, index=False)
            audit = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script": Path(__file__).name,
                "input_csv": str(input_csv),
                "output_csv": str(output_csv.resolve()),
                "raster": raster_info,
                "sampling": {
                    "coordinate_crs": "EPSG:4326",
                    "method": "nearest pixel containing the coordinate",
                    "latitude_column": args.latitude_column,
                    "longitude_column": args.longitude_column,
                    "band": 1,
                    "formula": "log1p(population_count_people_per_pixel)",
                    "allow_nodata": bool(args.allow_nodata),
                },
                "labels": label_audit,
            }
            audit_json.write_text(
                json.dumps(json_safe(audit), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"{input_csv.name}: {len(dataframe)} rows -> {output_csv.name}; "
                f"valid={label_audit['valid_samples']}, "
                f"nodata={label_audit['nodata_or_invalid_samples']}"
            )
            print(f"Audit: {audit_json}")


if __name__ == "__main__":
    main()
