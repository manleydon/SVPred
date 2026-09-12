# Street-View Value Audit

Core code for *Assessing Street-View Value in Urban Prediction: Spatial
Matching and Knowledge Transfer to Satellite-Only Models*.

The repository implements frozen DINOv2 feature extraction, spatial
counterfactual controls, strict label-isolated prediction distillation,
fixed-model distance replacement, and fold/raster integrity audits.

## Scope and data restrictions

The release contains code, dependencies, and schema-only examples. It does not
contain benchmark imagery, derived feature archives, checkpoints, predictions,
sample identifiers, coordinates, or experiment results. Obtain imagery under
the applicable provider terms and supply authorized local data before running
the pipeline.

The separate aggregate-results archive contains selected CityLens city-level
metrics and grouped-bootstrap summaries. It contains no sample-level records.

## Setup

Python 3.10 and a CUDA-compatible PyTorch installation are recommended.

```bash
python -m pip install -r requirements.txt
```

Run commands from the archive root. The scripts add their own directory to the
import path; do not change into `scripts/` first.

## Input schemas

### CityLens-style input

```text
data/example/citylens/
├─ Dataset/example_task.json
├─ satellite_image/ExampleCity/100_200.png
└─ street_view_image/Mapillary/image/100_200/
   ├─ view_01.jpg
   └─ view_02.jpg
```

Each task JSON item needs `area`, `reference`, and a non-empty `images` list.
The first path identifies the city; the local extractor reads the satellite
tile from `satellite_image/<city>/<area>.png`. Labels come from the JSON
`reference` field. At most ten street views are selected by sorted path and
mean-pooled after encoding. `--target-name` records metadata only.

```bash
python scripts/extract_citylens_features.py \
  --data-root data/example/citylens \
  --task-json Dataset/example_task.json \
  --target-name Population \
  --model /path/to/dinov2-base \
  --output artifacts/citylens_example.npz
```

### Haidian-style input

```text
data/example/haidian/
├─ Haidian.csv
├─ images/          satellite tiles, one per CSV row
└─ streetview/      street-view images referenced by streetview_files
```

Each CSV row needs `wgs84_lat`, `wgs84_lng`, `Filename`, `streetview_files`,
`population`, `log_Carbon`, and `BuildingHeight`. Semicolon-separated street
files are supported. Missing or corrupt image rows are reported and excluded.

```bash
python scripts/extract_dinov2_features.py \
  --csv data/example/haidian/Haidian.csv \
  --satellite-dir data/example/haidian/images \
  --street-dir data/example/haidian/streetview \
  --model /path/to/dinov2-base \
  --output artifacts/haidian_example.npz
```

An existing CityLens feature archive can be relabeled from another task JSON
without recomputing image features:

```bash
python scripts/relabel_citylens_features.py \
  --source-features artifacts/citylens_example.npz \
  --task-json Dataset/other_task.json \
  --target-name Healthcare \
  --output artifacts/citylens_healthcare_example.npz
```

## Reproduce the manuscript

The complete commands for the CityLens and Haidian configurations, strict KD
selection, bootstrap summaries, and integrity audits are in
[`docs/reproduction.md`](docs/reproduction.md).

## Main entry points

```bash
python scripts/run_citylens_minimal.py --help
python scripts/run_citylens_local_oracle.py --help
python scripts/run_strict_label_isolated_kd.py --help
python scripts/run_haidian_nested_spatial_controls.py --help
python scripts/run_haidian_distance_response_oracle.py --help
python scripts/audit_citylens_spatial_integrity.py --help
python scripts/relabel_haidian_worldpop_100m.py --help
python scripts/audit_worldpop100m_crossfold.py --help
```

## File roles

- **Feature extraction:** `extract_citylens_features.py`, `extract_dinov2_features.py`.
- **Experiment entry points:** the `run_citylens_*`, `run_haidian_*`, and
  `run_strict_label_isolated_kd.py` scripts.
- **Shared implementation:** `models.py`, `run_nested_spatial_kd.py`, and
  `release_utils.py`.
- **Relabeling, audits, and summaries:** the `relabel_*.py`, `audit_*.py`,
  `bootstrap_strict_label_isolated_kd.py`, and
  `summarize_strict_label_isolated_kd.py` scripts.

The code package covers the primary single-stage test-time conditions, strict
prediction distillation, fixed-model distance replacement, and fold/raster
audits. It does not include the separate Healthcare two-part sensitivity
launcher or its Brier/AUROC export.

## License

The code is released under the MIT License. Source imagery and derived feature
archives remain subject to the terms of their respective providers and are not
redistributed here.
