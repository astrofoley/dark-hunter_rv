## dark-hunter_rv (radial velocity pipeline)

This repo contains a **robust, echelle-aware RV measurement pipeline** targeting **~0.1 km/s** precision. It supports:

- **Order-by-order** (and optional **sub-chunk**) RV measurement.
- **Multiple RV techniques**:
  - stellar-mask **CCF** (with **multi-mask tournament**)
  - template **FFT correlation** (PHOENIX template bank; optional)
  - **strong-line (Balmer) centroiding** fallback
- Optional **order bias** correction via `bias_statistics.txt`.
- **Diagnostic plots** and **structured logging**.
- **Unit + smoke tests** (`pytest`).
- **Calibration workflow**: mask-only bias training, `bias_statistics.txt`, `method_rv_offsets.txt`, and a production batch driver (see **Documentation** below).

### Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/operations.md](docs/operations.md) | One-command calibration, production batch, env vars, cron, `--mask-only` / `--update` |
| [docs/rv_methods_evaluation.md](docs/rv_methods_evaluation.md) | Method validity, adopted-RV cascade, validation reports |
| [docs/contributing.md](docs/contributing.md) | Doc map, pre-PR checklist, **git commands to open a PR** |
| [docs/validation_playbook.md](docs/validation_playbook.md) | Validation campaigns |
| [docs/website.md](docs/website.md) | Public RV explorer deploy (`/var/www/html/darkhunter/rv`) |

### Quick start

Install dependencies (recommended in a clean environment):

```bash
python3 -m pip install -r requirements.txt
```

Run RVs on one or more spectra:

```bash
# From repo root
PYTHONPATH=. python3 -m darkhunter_rv.pipeline <files...> --instrument APF --teff 5500 --plots

# Or via wrapper
./measure_rvs.bash <files...> --instrument APF --teff 5500 --plots
```

Useful flags:

- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--plots` and `--plot-dir <dir>`
- `--continuum-mode split|spline|blaze|sinc_blaze|sinc_blaze_only` — default **`split`**: mask CCF uses per-order sinc² blaze only (`sinc_blaze_only`); template / strong-lines use blaze then spline (`sinc_blaze`). Requires `calibration/blaze_orders_apf.json` (build below). Escape hatches: `--no-blaze-continuum` or `--continuum-mode spline`.
- `--blaze-calibration <path>` — override blaze JSON (default: `calibration/blaze_orders_apf.json`)
- `--subchunks N` (split each order into N pixel chunks)
- **Default:** multi-method diagnostics (mask + template + strong lines) and **cascade** adopted RV (mask → template → strong; see `docs/rv_methods_evaluation.md`); `--no-run-all-methods` for legacy behavior
- `--mask-only` — stellar mask chunk RVs only (bias training; use with `--no-bias`); see `docs/operations.md`
- `--method-offsets-file` — global template/strong offsets vs mask (`method_rv_offsets.txt`)
- `--update` / `--force` — skip spectra whose diagnostics CSV is up to date (cron-friendly)
- `--max-chunk-err <km/s>` (skip chunks with invalid/huge errors)

**Batch workflows:** `python -m validation.run_calibration_setup` (bias + offsets + manifest) and `python -m validation.run_production_remaining` (process remaining spectra). Details in `docs/operations.md`.

Outputs:

- `output/<stem>_orders.txt` (per order/chunk)
- `output/<stem>_diagnostics.csv` (per chunk x method)
- `output/<stem>_tournament.csv` (mask tournament scores, if applicable)
- `output/<object>_summary.txt` (per-file RV summary)
- `plots/*` (if `--plots`)

### Bias and method offsets

**Recommended (current chunk `*_orders.txt` format):** use the automated calibration driver or `build_bias_set`:

```bash
python -m validation.run_calibration_setup --bias-list … --offset-list … --instrument APF
# or, manually:
python -m validation.build_bias_set --input-dir output --out-dir calibration/bias_build
# then copy calibration/bias_build/bias_statistics.txt to the repo root (APF default path)
```

**Legacy:** `python3 rv_bias.py` expects an older `*_orders.txt` layout (epoch/suborder columns). Prefer `build_bias_set` for outputs from the current pipeline.

**Method offsets** (template/strong vs mask): `python -m validation.compute_method_rv_offsets` (see `docs/operations.md`). Installed as `method_rv_offsets.txt` at repo root or via `DARKHUNTER_METHOD_OFFSETS_FILE`.

### Spectroscopic binary RV fitting (SB1, circular)

Given a pipeline summary file, fit a simple SB1 circular orbit (period fixed):

```bash
PYTHONPATH=. python3 -m darkhunter_rv.binary_cli output/<object>_summary.txt --period 3.4
```

This writes `<prefix>.sb1.txt` and `<prefix>.sb1.png`.

### Tests

```bash
PYTHONPATH=. python3 -m pytest -q
```

### Per-order blaze calibration (APF)

Build or refresh `calibration/blaze_orders_apf.json` from high-S/N campaign spectra:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python3 -m validation.build_blaze_calibration \
  --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \
  --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv
```

Hβ / clean-order diagnostic fits: `python -m validation.fit_hbeta_order_blaze --help`.

### Notes on execution environment

- In this Cursor environment, **NumPy/SciPy segfault inside the sandbox**.
  - Running Python commands **outside the sandbox** works.
  - If you hit `exit code 139`, rerun commands with appropriate permissions in Cursor.

### References

- KPF DRP reference repo (patterns for mask/CCF QA): `https://github.com/Keck-DataReductionPipelines/KPF-Pipeline`
