# Phase A baseline (RV precision framework)

Frozen goals and reference metrics for regression as Phases B–E improve chunking, bias, and weights.

## Goals

See [`goals.yaml`](goals.yaml):

- **Absolute gate:** APF vs literature pairs within 7 days, |ΔRV| < 1 km/s (ideally from `--no-bias` pipeline RVs).
- **Relative gate:** APF vs APF pairs within 7 days; track median/p90/RMS toward 0.1 km/s after debias.

## Run

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.rv_phase_a_baseline \
  --master calibration/literature_rv_master.csv \
  --summary-dir output \
  --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \
  --out-dir validation_output/rv_phase_a_baseline
```

For the canonical absolute gate (no bias correction), rerun the pipeline with `--no-bias` on overlap stars, then:

```bash
python -m validation.rv_phase_a_baseline \
  --summary-dir output \
  --no-bias-correction-applied \
  --out-dir validation_output/rv_phase_a_baseline_no_bias
```

## Outputs

| Path | Purpose |
|------|---------|
| `overlap_stars.csv` | Gaia IDs with both APF and literature epochs |
| `pair_candidates.csv` | All pairs within window, tagged by type |
| `absolute_gate_summary.csv` | Pass rate vs 1 km/s |
| `relative_gate_summary.csv` | Repeatability stats |
| `per_star_gates.csv` | Per-star breakdown |
| `plots/` | Diagnostic figures |
| `baseline_manifest.json` | Run metadata + metric hashes for regression |

## Regression

Compare `validation_output/rv_phase_a_baseline/baseline_manifest.json` to [`reference_manifest.json`](reference_manifest.json). Update the reference only when intentionally accepting a new baseline (e.g. after a planned pipeline change).
