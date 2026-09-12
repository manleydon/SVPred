"""Small shared utilities used by the public reproduction scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_CONTROLS = {
    "sat_supervised",
    "sat_parameter_matched",
    "sat_shuffled_real_sv",
    "sat_real_sv_teacher",
}


def parse_number_list(raw: str, cast: type) -> list:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("A comma-separated argument produced an empty list.")
    return sorted(set(values))


def load_reference(
    reference_dir: Path,
    sample_id: np.ndarray,
    city: np.ndarray,
    target: np.ndarray,
    spatial_seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and align a frozen CityLens outer-OOF reference run."""
    prediction_path = reference_dir / "outer_oof_predictions.csv"
    config_path = reference_dir / "run_config.json"
    reference = pd.read_csv(prediction_path)
    missing = REFERENCE_CONTROLS - set(reference["model"].unique())
    if missing:
        raise ValueError(f"Reference OOF file is missing controls: {sorted(missing)}")
    reference = reference[reference["model"].isin(REFERENCE_CONTROLS)].copy()
    baseline = reference[reference["model"] == "sat_supervised"].copy()
    if baseline["sample_id"].duplicated().any() or len(baseline) != len(sample_id):
        raise ValueError("Reference baseline must contain exactly one row per sample.")
    baseline["sample_id"] = baseline["sample_id"].astype(str)
    baseline = baseline.set_index("sample_id").loc[sample_id]
    if not np.array_equal(baseline["city"].astype(str).to_numpy(), city):
        raise ValueError("Feature archive and reference run have different city order.")
    if not np.allclose(baseline["y_true"].to_numpy(float), target):
        raise ValueError("Feature archive and reference run have different targets.")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("protocol") != "within_city_spatial":
            raise ValueError("The reference run must use within_city_spatial.")
        if int(config.get("spatial_seed", spatial_seed)) != spatial_seed:
            raise ValueError("--spatial-seed does not match the reference run.")
    return (
        reference,
        baseline["outer_fold"].to_numpy(int),
        baseline["outer_group"].astype(str).to_numpy(),
    )
