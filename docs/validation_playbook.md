# RV Validation Playbook

**See also:** [operations.md](operations.md) for **`run_calibration_setup`** / **`run_production_remaining`** and [rv_methods_evaluation.md](rv_methods_evaluation.md) for adopted-RV rules.

## Environment

- From the repo root, Python needs the package on the path. In **bash/zsh**: `export PYTHONPATH=.` then run `python3 validation/...`, or one shot: `env PYTHONPATH=. python3 validation/...`.
- **tcsh/csh** does not support `VAR=value command`; use `setenv PYTHONPATH .` then `python3 validation/...`, or `env PYTHONPATH=. python3 validation/...`.

## Commands

- **Full calibration (bias + method offsets + manifest):** `python -m validation.run_calibration_setup` (see [operations.md](operations.md)).
- Build bias set only:
  - `python3 validation/build_bias_set.py --input-dir output --out-dir validation_output/bias`
- Method consistency:
  - `python3 validation/evaluate_method_consistency.py --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/consistency`
- Broad-line benchmark:
  - `python3 validation/benchmark_broad_lines.py --out-dir validation_output/broad_line`
- Cool high-S/N mask precision (step 01; 0.1 km/s goal):
  - `python -m validation.benchmark_cool_precision --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/benchmark_cool_precision`
- **Phase A baseline** (overlap inventory + calibration gates; regression vs `calibration/phase_a_baseline/reference_manifest.json`):
  - `python -m validation.rv_phase_a_baseline --master calibration/literature_rv_master.csv --summary-dir output --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/rv_phase_a_baseline`
  - Absolute gate (APF vs literature, |ΔRV| < 1 km/s): use `--no-bias-correction-applied` after a `--no-bias` pipeline rerun on overlap stars.
  - Outputs: `overlap_stars.csv`, `pair_candidates.csv`, gate summaries, `plots/` (see `calibration/phase_a_baseline/README.md`).
- **Chunk residuals** (mask-applicable cool stars; per-object and sample bias plots):
  - `python -m validation.plot_chunk_residuals --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/chunk_residuals`
  - `--overlap-only` limits to phase-A overlap stars that pass the mask region cut.
  - Per object: `*_residuals_by_spectrum.png`, `*_chunk_weighted_mean.png`; sample: `sample_per_object_chunk_bias.png`.
  - Default clips per spectrum until convergence: 10σ leave-one-out (`--chunk-outlier-sigma 10`) and ±30 km/s from weighted mean (`--chunk-max-delta-kms 30`; `0` disables either). Excluded chunks shown as gray ×.
- Error model calibration:
  - `python3 validation/calibrate_error_model.py --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/error_model`
- Full campaign report:
  - `python3 validation/run_campaign.py --orders-dir output --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/campaign`
- Legacy vs new pipeline (APF + Gaia):
  - `env PYTHONPATH=. python3 validation/diagnose_legacy_campaign.py --data-dir ../data --legacy-output-dir ../output --new-output-dir validation_output/pipeline_rerun --report-dir validation_output/diagnose_legacy --spectrum-glob 'Gaia_DR3_1702*.txt' --run-pipeline --query-gaia --dump-gaia-json -- --instrument APF --run-all-methods --log-level ERROR`
  - Omit `--no-bias` to apply repo [`bias_statistics.txt`](bias_statistics.txt) (match legacy if it was debiased). Add `--no-bias` only for an explicit no-debias comparison.
  - Pipeline flags go **after a lone `--`**.
  - **Multi-star:** use a broad glob (e.g. `Gaia_DR3_*.txt`) and **`--multi-star`**; reports go under `report-dir/<source_id>/`. Optional `--min-epochs 10` and `--write-combined-csv`.
- Interpretation plots + text (after diagnose, or on existing CSVs):
  - `env PYTHONPATH=. python3 validation/legacy_interpretation_report.py --report-dir validation_output/diagnose_legacy --pipeline-summary validation_output/pipeline_rerun/Gaia_DR3_<id>_summary.txt --legacy-summary ../output/<id>_summary.txt`

## Key outputs

- `bias/bias_statistics.txt`, `bias/bias_by_chunk.csv`
- `consistency/method_pair_offsets.csv`, `consistency/method_trends.csv`
- `broad_line/broad_line_summary.csv`, `docs/broad_line_method.md`
- `error_model/systematic_floors.csv`, `error_model/coverage_report.csv`
- `campaign/validation_report.json`
- `diagnose_legacy/exposure_comparison.csv` (includes `gaia_source_id` when multi-star), `method_exposure_summary.csv`, `method_pair_stats.csv`, `by_teff_bin.csv`, optional `gaia_query_*.json`, `interpretation_summary.txt`, `rv_vs_mjd.png`, `methods_heatmap.png`, `delta_method_vs_teff.png`, `delta_rv_histogram.png`, etc.

## Suggested acceptance thresholds

- Cool-star method pair median offsets: `|offset| < 0.1 km/s`
- Chunk rejection rates should be stable by night/instrument (no pathological swings)
- Calibrated 1-sigma coverage target: roughly `0.60-0.75`
- Calibrated 2-sigma coverage target: roughly `0.90-0.98`

## Interpretation notes

- If method offsets are coherent with Teff or mask-line count, use those features in post-hoc method trust regions.
- If error coverage is under-dispersed, increase systematic floor terms per method/instrument/chunk family.
- For broad-line stars, use the benchmark recommendation from `docs/broad_line_method.md`.
