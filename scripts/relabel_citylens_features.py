"""Reuse a CityLens feature archive with labels from another global task JSON.

Only samples present in both the source archive and target task are retained.
Feature arrays are copied without recomputing DINO embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument(
        "--minimum-target",
        type=float,
        default=None,
        help="Keep targets strictly greater than this value (e.g. 0 for healthcare).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.source_features, allow_pickle=False)
    items = json.loads(args.task_json.read_text(encoding="utf-8"))
    areas = [str(item["area"]) for item in items]
    if len(set(areas)) != len(areas):
        raise ValueError("Task JSON contains duplicate area identifiers.")
    labels = {
        area: float(item["reference"])
        for area, item in zip(areas, items, strict=True)
    }
    if args.minimum_target is not None:
        labels = {
            area: value
            for area, value in labels.items()
            if value > args.minimum_target
        }
    sample_id = source["sample_id"].astype(str)
    keep = np.asarray([area in labels for area in sample_id], dtype=bool)
    if not keep.any():
        raise ValueError("No source samples occur in the target task JSON.")
    payload: dict[str, np.ndarray] = {}
    for key in source.files:
        values = source[key]
        if key in {"target", "target_name"}:
            continue
        if values.ndim >= 1 and values.shape[0] == len(sample_id):
            payload[key] = values[keep]
        else:
            payload[key] = values
    kept_ids = sample_id[keep]
    payload["target"] = np.asarray([labels[area] for area in kept_ids], dtype=np.float32)
    payload["target_name"] = np.asarray([args.target_name], dtype=np.str_)
    zero_count = int(np.count_nonzero(payload["target"] == 0))
    positive_count = int(np.count_nonzero(payload["target"] > 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved: {args.output}")
    print(f"Source samples: {len(sample_id)} | retained: {len(kept_ids)}")
    print(f"Retained zero: {zero_count} | retained positive: {positive_count}")
    if args.minimum_target is not None:
        print(f"Filter: target > {args.minimum_target:g}")
    print(
        f"Target raw range: {payload['target'].min():.6g} .. "
        f"{payload['target'].max():.6g}"
    )


if __name__ == "__main__":
    main()
