# Website tables

Required files (loaded by `script.js`):

| File | Required |
|------|----------|
| `data.csv` | Yes |
| `keck_targets.csv` | Yes (may be empty header + rows) |

Optional:

| File | Purpose |
|------|---------|
| `simbad_gaia_ids.csv` | SIMBAD links in Gaia ID column |

Seed from the legacy site once:

```bash
bash scripts/bootstrap_website_tables.sh
```
