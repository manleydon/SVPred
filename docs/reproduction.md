# Reproduction guide

Run every command from the archive root. Replace illustrative paths with
authorized local paths. The benchmark imagery, feature archives, checkpoints,
predictions, labels, and coordinates are not part of this release.

Feature extraction must be completed before model training. The CityLens
reference run must be completed before the corresponding local-oracle or KD
run.

## CityLens

### Extract features

Population uses the released Population task JSON:

```bash
python scripts/extract_citylens_features.py \
  --data-root /path/to/CityLens-data \
  --task-json Dataset/all_global_pop_task_all.json \
  --target-name Population \
  --model /path/to/dinov2-base \
  --output artifacts/citylens_population.npz
```

For Healthcare, use the task JSON whose `reference` field contains the full
Healthcare-overlap labels. `--target-name` records metadata only; it does not
change the labels.

```bash
python scripts/extract_citylens_features.py \
  --data-root /path/to/CityLens-data \
  --task-json /path/to/healthcare_task.json \
  --target-name Healthcare \
  --model /path/to/dinov2-base \
  --output artifacts/citylens_healthcare_all.npz
```

The extractor selects at most ten street views by sorted path, validates their
decoding, and does not backfill a failed selection.

### Reference controls

Population uses one seed-42 reference for all five KD seeds:

```bash
python scripts/run_citylens_minimal.py \
  --features artifacts/citylens_population.npz \
  --output-dir runs/citylens_population_within_city_seed42 \
  --target-transform log1p \
  --spatial-seed 42 --seed 42
```

Healthcare uses a reference run for each model seed:

```bash
for seed in 41 42 43 44 45; do
  python scripts/run_citylens_minimal.py \
    --features artifacts/citylens_healthcare_all.npz \
    --output-dir runs/citylens_healthcare_all_within_city_seed${seed} \
    --target-transform log1p \
    --spatial-seed 42 --seed ${seed}
done
```

The retrained Minimum-cost Intermediate condition is generated from the same
reference runs:

```bash
for seed in 41 42 43 44 45; do
  python scripts/run_citylens_local_oracle.py \
    --features artifacts/citylens_population.npz \
    --reference-dir runs/citylens_population_within_city_seed42 \
    --output-dir runs/citylens_population_local_oracle_seed${seed} \
    --target-transform log1p --spatial-seed 42 --seed ${seed}

  python scripts/run_citylens_local_oracle.py \
    --features artifacts/citylens_healthcare_all.npz \
    --reference-dir runs/citylens_healthcare_all_within_city_seed${seed} \
    --output-dir runs/citylens_healthcare_all_local_oracle_seed${seed} \
    --target-transform log1p --spatial-seed 42 --seed ${seed}
done
```

### Strict prediction distillation

Population:

```bash
python scripts/run_strict_label_isolated_kd.py \
  --profile citylens-population \
  --output-root runs/strict_kd \
  --seeds 41,42,43,44,45 \
  --kd-weight-grid 0,0.25,0.5,1,2,4,8,16,32,64 \
  --citylens-population-features artifacts/citylens_population.npz \
  --citylens-population-reference-root runs \
  --execute --skip-existing
```

Healthcare:

```bash
python scripts/run_strict_label_isolated_kd.py \
  --profile citylens-healthcare \
  --output-root runs/strict_kd \
  --seeds 41,42,43,44,45 \
  --kd-weight-grid 0,0.25,0.5,1,2,4,8,16,32,64 \
  --citylens-healthcare-features artifacts/citylens_healthcare_all.npz \
  --citylens-healthcare-reference-root runs \
  --execute --skip-existing
```

The wrapper fixes the spatial and shuffled-assignment seeds at 42 while varying
the model seed. Add `--no-progress` only when non-interactive logs are needed.

### CityLens integrity audit

After a primary run, the audit checks cross-fold image hashes and nearest-fold
separation:

```bash
python scripts/audit_citylens_spatial_integrity.py \
  --oof runs/citylens_population_within_city_seed42/outer_oof_predictions.csv \
  --satellite-root /path/to/CityLens-data/satellite_image \
  --street-root /path/to/CityLens-data/street_view_image/Mapillary/image \
  --output-dir runs/citylens_population_integrity
```

## Haidian

### Relabel and encode the source splits

Keep the validated train and test CSVs separate for the WorldPop cell audit:

```bash
python scripts/relabel_haidian_worldpop_100m.py \
  --input-csv /path/to/Haidian_train.csv /path/to/Haidian_test.csv \
  --raster /path/to/chn_pop_2020_CN_100m_R2025A_v1.tif \
  --output-dir artifacts/haidian_worldpop100m \
  --raster-checksum

python scripts/extract_dinov2_features.py \
  --csv artifacts/haidian_worldpop100m/Haidian_train_worldpop100m.csv \
  --satellite-dir /path/to/haidian/satellite \
  --street-dir /path/to/haidian/streetview \
  --model /path/to/dinov2-base \
  --output artifacts/haidian_train.npz

python scripts/extract_dinov2_features.py \
  --csv artifacts/haidian_worldpop100m/Haidian_test_worldpop100m.csv \
  --satellite-dir /path/to/haidian/satellite \
  --street-dir /path/to/haidian/streetview \
  --model /path/to/dinov2-base \
  --output artifacts/haidian_test.npz
```

Audit the two extracted feature archives against the same raster and spatial
fold construction:

```bash
python scripts/audit_worldpop100m_crossfold.py \
  --features artifacts/haidian_train.npz artifacts/haidian_test.npz \
  --raster /path/to/chn_pop_2020_CN_100m_R2025A_v1.tif \
  --output-dir runs/haidian_worldpop100m_crossfold \
  --outer-folds 5 --spatial-blocks 25 --spatial-seed 42
```

### Nested spatial controls

Concatenate the two validated relabeled CSVs before the nested run:

```bash
python -c "import pandas as pd; pd.concat([pd.read_csv('artifacts/haidian_worldpop100m/Haidian_train_worldpop100m.csv'), pd.read_csv('artifacts/haidian_worldpop100m/Haidian_test_worldpop100m.csv')], ignore_index=True).to_csv('artifacts/haidian_worldpop100m/Haidian_all_worldpop100m.csv', index=False)"

python scripts/extract_dinov2_features.py \
  --csv artifacts/haidian_worldpop100m/Haidian_all_worldpop100m.csv \
  --satellite-dir /path/to/haidian/satellite \
  --street-dir /path/to/haidian/streetview \
  --model /path/to/dinov2-base \
  --output artifacts/haidian_population100m.npz
```

One strict seed can be run directly:

```bash
python scripts/run_haidian_nested_spatial_controls.py \
  --features artifacts/haidian_population100m.npz \
  --output-dir runs/haidian_population_strict_kd_seed42 \
  --targets population \
  --outer-folds 5 --inner-folds 4 --spatial-blocks 25 \
  --kd-weight-grid 0,0.25,0.5,1,2,4,8,16,32,64 \
  --lambda-pred 0.5 \
  --spatial-seed 42 --shuffle-seed 42 --seed 42 \
  --strict-label-isolation
```

All five Haidian model seeds can be launched by the wrapper:

```bash
python scripts/run_strict_label_isolated_kd.py \
  --profile haidian \
  --output-root runs/strict_kd \
  --seeds 41,42,43,44,45 \
  --kd-weight-grid 0,0.25,0.5,1,2,4,8,16,32,64 \
  --haidian-features artifacts/haidian_population100m.npz \
  --execute --skip-existing
```

### Fixed-model distance replacement

The following command reproduces one model seed of the distance analysis. The
manuscript summary repeats it for seeds 41--45 while keeping matching seed 42:

```bash
python scripts/run_haidian_distance_response_oracle.py \
  --features artifacts/haidian_population100m.npz \
  --output-dir runs/haidian_population_distance_response_seed42 \
  --target population \
  --distance-bands-m 0-500,500-1000,1000-2000,2000-5000,5000-inf \
  --outer-folds 5 --spatial-blocks 25 \
  --spatial-seed 42 --matching-seed 42 --seed 42 \
  --bootstrap-repetitions 2000 --skip-retrained-oracles
```

## Summaries and audits

After all strict runs are complete, summarize model-seed comparisons and, when
available, the less-isolated sensitivity:

```bash
python scripts/summarize_strict_label_isolated_kd.py \
  --strict-root runs/strict_kd \
  --less-isolated-root runs/less_isolated_kd \
  --output-dir runs/strict_kd/summary
```

Generate the grouped strict-KD intervals:

```bash
python scripts/bootstrap_strict_label_isolated_kd.py \
  --strict-root runs/strict_kd \
  --output-dir runs/strict_kd/summary \
  --repetitions 2000 --seed 20260909
```

Training outputs include `run_config.json`, outer-OOF predictions, pooled
summaries, fold metrics, and relevant diagnostic files. The two Healthcare
two-part sensitivity outputs are not generated by this core package.

Selected CityLens aggregate tables are provided in `results/`. They are derived
from the authors' outer-OOF records and do not replace the restricted inputs or
sample-level outputs.
