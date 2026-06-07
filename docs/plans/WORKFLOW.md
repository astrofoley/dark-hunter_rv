# RV pipeline step workflow

**Repo-visible step plans:** [steps/](steps/) (mirror of `.cursor/plans/rv-pipeline/steps/` for Markdown preview and GitHub browsing). Update both when a step changes.

GitHub **issues** and **branches** track shared progress.

## A. Start a step

1. Read [INDEX.md](INDEX.md) — confirm dependencies are completed.
2. Open `steps/NN-<slug>.md`; set frontmatter `status: in_progress`.
3. Comment on the GitHub issue with the branch name.
4. From repo root:

   ```bash
   cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
   git checkout main && git pull
   git checkout -b step/NN-<slug>
   ```

5. Work through implementation task checkboxes in the step plan.

## B. Finish a step (merge PR)

1. **Step md:** `status: completed`; check boxes; record issue # and merged branch (both `docs/plans/steps/` and `.cursor/plans/rv-pipeline/steps/`).
2. **Master plan** `.cursor/plans/rv_pipeline_master_plan_8447f2cd.plan.md`:
   - Set matching frontmatter `todos[].status: completed`
   - Update phase / “What has been implemented” if needed
3. **INDEX.md** (repo + `.cursor/plans/rv-pipeline/`): update status, issue #, branch, merge date.
4. **Legacy plans** (listed in step `related_legacy_plans`): mark relevant todos or add “superseded by step NN”.
5. **Repo docs** (`repo_docs_to_update` in step): add commands only when user-facing behavior changes.
6. Close GitHub issue with merge summary.

## C. Branch convention

- Prefix: `step/`
- One logical step per PR; sub-branches `step/02a-...`, `step/02b-...` for large steps
- PR title: `step/NN: <title>`
- PR body: `Closes #<issue>` + plan path `docs/plans/steps/NN-<slug>.md`

## D. Cursor session

Attach `@docs/plans/steps/NN-<slug>.md` (preview) or `@.cursor/plans/rv-pipeline/steps/NN-<slug>.md` (canonical local).

## E. Propagation map (legacy plans on completion)

| Step | Legacy `.cursor/plans/` |
|------|-------------------------|
| 01 | `rv_pipeline_roadmap_3a7b3787`, `rv_methods_evaluation_plan_fcd09d94` |
| 02 | `rv_pipeline_roadmap_3a7b3787`, `rv_mismatch_diagnosis_24a11615`, `pr_repo_organization_326c519c` |
| 03 | `rv_method_fusion_plan_b4fd46f0`, `rv_methods_evaluation_plan_fcd09d94`, `three_rv_methods_e1b72701` |
| 04 | `legacy_plot_and_ccf_qc_2f3b70cd`, `gaia_cache_and_ccf_diagnostics_2e731512` |
| 05 | `rv_pipeline_roadmap_3a7b3787`, `pr_repo_organization_326c519c` |
| 06 | `template_grid_and_hβ_rv_dff5dfce`, `three_rv_methods_e1b72701` |
| 07 | `rv_pipeline_roadmap_3a7b3787` |
| 08 | `pipeline_legacy_diagnostics_e98cdb98` |
