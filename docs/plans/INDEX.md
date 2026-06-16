# RV pipeline step tracker (GitHub)

**Step plans (rendered preview in IDE):** [steps/](steps/) — open any `.md` and use **Markdown: Open Preview** (Cmd+Shift+V).

Orchestrator (local): `.cursor/plans/rv_pipeline_master_plan_8447f2cd.plan.md`

Workflow: [WORKFLOW.md](WORKFLOW.md)

## Per-method precision lanes (current focus)

| Lane | Steps | Status |
|------|-------|--------|
| **mask_ccf** | 01, 02a, 09 | Defaults → `subchunks_8`; **bias rebuild running** |
| **template_fft** | 10 | **Active** — baseline overlap, then knob tuning |
| **strong_lines** | 06 | After template baseline |
| **fusion / adoption** | 03, 04 | After lanes above |

| Step | Plan | Status | Issue | Branch(es) | Merged |
|------|------|--------|-------|------------|--------|
| 00 Literature RV master | [steps/00-literature-rv-master.md](steps/00-literature-rv-master.md) | completed | [#37](https://github.com/astrofoley/dark-hunter_rv/issues/37) | `step/00-literature-rv-master` | 2026-06-07 ([#46](https://github.com/astrofoley/dark-hunter_rv/pull/46)) |
| 01 Benchmark cool precision | [steps/01-benchmark-cool-precision.md](steps/01-benchmark-cool-precision.md) | in_progress | [#38](https://github.com/astrofoley/dark-hunter_rv/issues/38) | `step/01-benchmark-cool-precision` | — |
| 02 Chunk weights / subchunks | [steps/02-chunk-weights-subchunks.md](steps/02-chunk-weights-subchunks.md) | in_progress (02a done) | [#39](https://github.com/astrofoley/dark-hunter_rv/issues/39) | `step/02a-subchunk-study`, `step/02b-trust-weights-stack` | — |
| 03 Method fusion / coverage | [steps/03-method-fusion-coverage.md](steps/03-method-fusion-coverage.md) | pending | [#40](https://github.com/astrofoley/dark-hunter_rv/issues/40) | `step/03-method-fusion-coverage` | — |
| 04 Adopted-RV match plots | [steps/04-adopted-rv-match-plots.md](steps/04-adopted-rv-match-plots.md) | pending | [#41](https://github.com/astrofoley/dark-hunter_rv/issues/41) | `step/04-adopted-rv-match-plots` | — |
| 05 Short-pair + epoch CCF | [steps/05-short-pair-epoch-ccf.md](steps/05-short-pair-epoch-ccf.md) | pending | [#42](https://github.com/astrofoley/dark-hunter_rv/issues/42) | `step/05a-short-pair-calibration`, `step/05b-epoch-ccf-consistency` | — |
| 06 Strong-line line list | [steps/06-strong-line-line-list.md](steps/06-strong-line-line-list.md) | pending | [#43](https://github.com/astrofoley/dark-hunter_rv/issues/43) | `step/06-strong-line-line-list` | — |
| 07 SB2 search | [steps/07-sb2-search.md](steps/07-sb2-search.md) | pending | [#44](https://github.com/astrofoley/dark-hunter_rv/issues/44) | `step/07a-sb2-detection`, `step/07b-sb2-reporting` | — |
| 08 External RV cross-check | [steps/08-external-rv-crosscheck.md](steps/08-external-rv-crosscheck.md) | pending | [#45](https://github.com/astrofoley/dark-hunter_rv/issues/45) | `step/08-external-rv-crosscheck` | — |
| 09 CCF RV estimator (mask) | [steps/09-ccf-rv-estimator.md](steps/09-ccf-rv-estimator.md) | complete (`gauss_offset`) | — | — | — |
| 10 Template FFT precision | [steps/10-template-fft-precision.md](steps/10-template-fft-precision.md) | **in_progress** | (create) | `step/10-template-fft-precision` | — |

Update this file when an issue closes or a step status changes. Keep in sync with `.cursor/plans/rv-pipeline/INDEX.md`.
