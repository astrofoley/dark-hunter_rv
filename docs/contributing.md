# Contributing and pull requests

## Documentation map

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Install, quick start, flags overview, tests |
| [operations.md](operations.md) | Calibration (`run_calibration_setup`), production (`run_production_remaining`), env vars, cron |
| [rv_methods_evaluation.md](rv_methods_evaluation.md) | Method validity, adopted-RV cascade, overlap reports |
| [validation_playbook.md](validation_playbook.md) | Validation / campaign workflows |
| [plans/INDEX.md](plans/INDEX.md) | RV pipeline GitHub step tracker (issues #37–#45) |
| [broad_line_method.md](broad_line_method.md) | Broad-line / Hβ methodology notes |

**Validation CLIs** (run from repo root with `PYTHONPATH=.` or `python -m …` as shown):

- `validation.run_calibration_setup` — one-shot bias + method offsets + `calibration/manifest.json`
- `validation.run_production_remaining` — batch process spectra, skip offset-training set when diagnostics exist
- `validation.build_bias_set` — `bias_statistics.txt` from current `*_orders.txt` chunk format
- `validation.compute_method_rv_offsets` — `method_rv_offsets.txt` (mask = truth)
- `validation.rv_method_diagnostics_report`, `validation.rv_method_overlap_report` — method comparison plots/tables

**Pipeline flags** (see `python -m darkhunter_rv.pipeline --help`):

- Default: multi-method diagnostics; adopted RV = mask → template → strong cascade (`method_evaluation`, `method_regions`)
- `--mask-only` — stellar mask chunk RVs only (bias training; no PHOENIX / template / strong)
- `--no-bias` — do not apply `bias_statistics.txt` (use when building biases)
- `--no-run-all-methods` — legacy single-method behavior
- `--method-offsets-file`, `--update`, `--force` — offsets path and incremental reprocessing

## Before you open a PR

1. Run tests from repo root (use full permissions if NumPy segfaults in a restricted sandbox):

   ```bash
   cd /path/to/dark-hunter_rv
   PYTHONPATH=. python3 -m pytest -q
   ```

2. Do **not** commit generated science outputs: `output/`, `plots/`, `validation_output/` are gitignored. Do not add large binaries or per-machine paths in committed lists.

3. Prefer **absolute or env-based paths** in user-edited list files; only example lists belong in the repo if small.

## Commands to open a pull request

Replace the branch name and commit message as needed.

```bash
cd /path/to/dark-hunter_rv

# Start from latest main
git fetch origin
git checkout main
git pull origin main

# Feature branch
git checkout -b feature/pipeline-calibration-and-docs

# Stage and commit (review with `git status` / `git diff --stat` first)
git add -A
git status
git commit -m "Pipeline: calibration workflow, cascade adoption, docs, and tests"

# Push and create PR (GitHub CLI)
git push -u origin feature/pipeline-calibration-and-docs
gh pr create --title "Pipeline calibration workflow, cascade RV adoption, documentation" \
  --body "## Summary
- Multi-method default with mask→template→strong adopted-RV cascade and optional method offsets.
- \`--mask-only\` bias training; \`run_calibration_setup\` / \`run_production_remaining\`; manifest JSON.
- \`--update\` / \`--force\` for incremental runs; expanded docs and tests.

## Test plan
- [ ] \`PYTHONPATH=. pytest -q\` locally
"

```

If you do not use `gh`, push the branch and open a PR in the GitHub web UI:

```bash
git push -u origin feature/pipeline-calibration-and-docs
```

Then visit `https://github.com/astrofoley/dark-hunter_rv/compare` and compare your branch to `main`.

## Legacy `rv_bias.py`

The repository root script `rv_bias.py` targets an older `*_orders.txt` column layout. Current chunk-style outputs are aggregated by `python -m validation.build_bias_set` (used by `run_calibration_setup`). See [operations.md](operations.md).
