# RV methods evaluation (binaries, no single true RV)

**See also:** [operations.md](operations.md) (calibration and production drivers), [contributing.md](contributing.md).

This note turns the evaluation plan into **operational definitions** and points to the code that implements them.

## Problem framing

Many targets are **binaries or candidate binaries**; there is no unique heliocentric RV to use as absolute truth. We instead:

1. Measure **inter-method agreement** where multiple methods simultaneously pass validity (overlap).
2. Summarize **bias-like structure** (e.g. median of mask − template) in bins of **Teff**, **S/N proxy**, and optionally **Gaia** parameters (RUWE, [M/H], log g) once star summaries exist.
3. Choose an **adopted exposure RV** from among valid methods using a **transparent rule** (see below).

## Validity (per method, per exposure)

Implemented in `darkhunter_rv.method_evaluation.exposure_method_flags`:

| Method | Valid when |
|--------|------------|
| `mask_ccf` | `_weighted_method_rv_from_rows` returns finite RV and σ>0 after chunk QC and minimum chunk count (`MIN_MASK_CCF_CHUNKS_FOR_STACK`). |
| `template_fft` | Same pattern for template chunks (`MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK`). |
| `strong_lines` | Single `chunk_key=all` row (Voigt+Lorentz **Hβ** centroid today) with `qc_pass`, finite RV/err, σ in (0, 1e27]. |

**Overlap** for an exposure: `n_methods_valid ≥ 2`. Pairwise deltas (e.g. mask − template) are only defined when **both** methods in the pair are valid.

**S/N proxy:** median of `ccf_peak_snr` over per-order `mask_ccf` rows (excluding the synthetic `all` row).

## Adopted RV rule

`darkhunter_rv.method_evaluation.recommend_adopted_rv` (used by the overlap report and, with multi-method pipeline runs, the main pipeline):

- **Cascade** (preference order): `mask_ccf` → `template_fft` → `strong_lines`.
- Each method must be **valid** (see table above) and **region-applicable** (`darkhunter_rv.method_regions`, same cuts as residual plots: Teff and log₁₀ median mask CCF peak S/N).
- Use the first method in that order with σ ≤ `ADOPTED_CASCADE_MAX_SIGMA_KMS` (default matches comparison-report cap, env `DARKHUNTER_ADOPTED_MAX_SIGMA_KMS`).
- If none meet the σ cut, adopt the **first** applicable valid method in order and keep its (possibly large) σ.
- If none apply: adopted fields are empty / NaN.

Optional **method offsets** (`method_rv_offsets.txt`): add calibrated shifts to template and strong-line RVs before adoption (mask = truth reference). See `docs/operations.md`.

**Note:** regions use mask CCF S/N; a cheaper S/N-only proxy for early gating is future work (methods still run in v1).

## Reports and CLI

| Artifact | Producer |
|----------|----------|
| `method_comparison_per_exposure.csv`, Teff residual plots | `python -m validation.rv_method_diagnostics_report` |
| `overlap_enriched_per_exposure.csv`, binned CSVs, overlap histogram, residual vs log₁₀(S/N) | `python -m validation.rv_method_overlap_report` |

Overlap report arguments:

- `--diagnostics-glob` — same pattern as the Teff report.
- `--gaia-summary-dir` — directory containing `Gaia_DR3_*_summary.txt` (fills `gaia_ruwe`, `gaia_mh`, `gaia_logg` when `[GAIA METADATA]` parses).
- `--min-bin-count` — minimum exposures per bin for binned median/MAD tables (default 15).

Campaign shortcut:

```bash
python -m validation.run_first_epoch_campaign ... --method-teff-report --overlap-report
```

## QC and parameter space

- **Hard vs soft QC:** pipeline row flag `qc_pass` is a hard gate for mask/template stack rows and for the exposure-level `strong_lines` row. Further cuts (e.g. RUWE, NSS flags) belong in analysis layers or documented campaign filters — not silently inside the adoption helper.
- **Continuum A/B** and a full **literature** pass are **not** implemented here; treat them as follow-on experiments once overlap statistics are stable.

## Order PDFs when two RVs disagree

Same tool as legacy outliers: ``python -m validation.plot_legacy_outlier_orders``. **Legacy:** ``--comparison-csv`` + ``--max-legacy-err``. **Two valid methods:** ``--overlap-csv`` pointing at ``overlap_enriched_per_exposure.csv`` and ``--threshold-kms`` (e.g. 50); one PDF per exposure per disagreeing pair, each page an echelle order with the mask schematic at both velocities.

## Tests

`tests/test_method_evaluation.py` covers stack rules, QC on single-row methods, adoption ordering, and the S/N median helper.
