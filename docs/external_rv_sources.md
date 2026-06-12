# External RV catalog sources

Star summaries store third-party spectroscopic RVs in `[EXTERNAL RV DATA]`. **All values are solar-system barycentric (km/s)** after ingest, using each survey's observation MJD, observatory site, and the target RA/Dec from `[GAIA METADATA]`.

Heliocentric catalog values are converted with `astropy.coordinates.SkyCoord.radial_velocity_correction` (see [`darkhunter_rv/rv_frame.py`](../darkhunter_rv/rv_frame.py)).

## Batch update

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python3 scripts/update_summary_external_rvs.py --sources galah,apogee,desi
python3 scripts/update_summary_external_rvs.py --star-id 1551542027851147904 --sources galah,apogee,ges
```

DESI-only (legacy script):

```bash
python3 scripts/update_summary_desi_rvs.py --star-id 1551542027851147904
```

## Catalog reference

| Source | Access | Native frame | Site key | Epoch column | Typical σ |
|--------|--------|--------------|----------|--------------|-----------|
| GALAH DR3 | VizieR `J/MNRAS/506/150/rv` + `stars` (`GaiaEDR3`) | Heliocentric | `GALAH_AAT` | `MJDlocal` | ~0.1 km/s |
| APOGEE DR17 | VizieR `III/286/catalog` + `allvis` (`VHelio`, `Tel`) | Barycentric | `APOGEE_APO` / `APOGEE_LCO` | `MJD` | ~0.1 km/s |
| Gaia-ESO DR5.1 | ESO archive only (not on VizieR TAP) | Barycentric | `GES_VLT` | `DATE_OBS` | 0.2–0.4 km/s |
| DESI MWS DR1 | NOIRLab Data Lab `desi_dr1.mws` | Heliocentric | `DESI` | `min_mjd`/`max_mjd` | ~1 km/s |
| LAMOST LRS DR9 | Gaia `external.lamost_dr9_lrs` | Heliocentric (`z`) | `LAMOST` | `obsdate` | variable |
| LAMOST MRS DR9 | Gaia `external.lamost_dr9_mrs` | Barycentric (`rv_br1`) | `LAMOST` | `obsdate` | ~1 km/s |
| RAVE DR6 | Gaia `external.ravedr6` | Heliocentric | `RAVE` | `rave_obs_id` date | ~1 km/s |

**Not used:** Gaia DR3 mission-mean `radial_velocity` (no per-epoch MJD; may blend epochs).

## Flag column

Provenance is recorded in the `Flag/ID` field, e.g. `conv=helio→bary`, `frame=bary-native`, survey ids, cone separation.

## Quality cuts at ingest

All catalog rows with a finite RV and valid MJD are written to the summary (including large `rv_err`). Optional `--max-rv-err KM_S` on the batch script filters at download time; omit it to keep everything.

Rows without valid MJD (≥ 40000) or coordinates are still skipped (needed for barycentric conversion and orbit fits).
