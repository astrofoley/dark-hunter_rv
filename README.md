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
- `--continuum-mode spline|blaze`
- `--subchunks N` (split each order into N pixel chunks)
- `--run-all-methods` (emit diagnostics for all methods)
- `--max-chunk-err <km/s>` (skip chunks with invalid/huge errors)

Outputs:

- `output/<stem>_orders.txt` (per order/chunk)
- `output/<stem>_diagnostics.csv` (per chunk x method)
- `output/<stem>_tournament.csv` (mask tournament scores, if applicable)
- `output/<object>_summary.txt` (per-file RV summary)
- `plots/*` (if `--plots`)

### Bias estimation

After you have many `output/*_orders.txt` files, build `bias_statistics.txt`:

```bash
python3 rv_bias.py --input-dir output --output bias_statistics.txt
```

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

### Notes on execution environment

- In this Cursor environment, **NumPy/SciPy segfault inside the sandbox**.
  - Running Python commands **outside the sandbox** works.
  - If you hit `exit code 139`, rerun commands with appropriate permissions in Cursor.

### References

- KPF DRP reference repo (patterns for mask/CCF QA): `https://github.com/Keck-DataReductionPipelines/KPF-Pipeline`
