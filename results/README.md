# Street-View Value Audit: Public Aggregate Results

This directory contains non-sensitive aggregate tables supporting the manuscript
*Assessing Street-View Value in Urban Prediction: Spatial Matching and Knowledge
Transfer to Satellite-Only Models*.

Included files:

- `citylens_city_metrics_five_seed.csv`: CityLens city-level (R^2), RMSE,
  and MAE for the primary test-time conditions, by task, seed, city, and
  model.
- `citylens_kd_city_metrics_five_seed.csv`: CityLens city-level metrics for
  the strict label-isolated prediction-distillation controls.
- `citylens_grouped_bootstrap_intervals.csv`: CityLens city and city--block
  bootstrap contrasts and conditional interval endpoints for the primary
  test-time contrasts in both tasks. Strict KD grouped intervals are included
  for the Healthcare overlap task in this release.

These files contain city names, sample counts, model labels, aggregate metrics,
and resampling summaries only. They contain no sample identifiers, coordinates,
image paths, imagery, features, checkpoints, predictions, or raw targets.
The Healthcare overlap rows use all 2,204 labels; the positive-label subset is
not substituted for the primary task. The KD rows use the strict
label-isolated configurations used for the primary comparison.
They are provided to make the reported city-level and grouped-inference
summaries directly checkable without redistributing restricted source data.

This directory contains CityLens aggregate results only; Haidian aggregate
tables and sample-level outputs are not included. The values are derived from
the authors' outer-OOF records and are not a replacement for the full input
data or feature archives. The code and schemas are provided in the repository's
root `scripts/`, `data/example/`, and `docs/` directories.
