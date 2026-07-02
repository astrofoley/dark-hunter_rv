from pathlib import Path

from darkhunter_rv.summary_paths import discover_summary_path


def test_discover_summary_prefers_richer_pipeline_file(tmp_path: Path) -> None:
    gid = "77413727493690112"
    flat = tmp_path / f"Gaia_DR3_{gid}_summary.txt"
    flat.write_text(
        "### STAR SUMMARY ###\n\n[GAIA METADATA]\nSource_ID: {}\n\n[PIPELINE RESULTS]\n# empty\n".format(
            gid
        )
    )
    nested_dir = tmp_path / f"Gaia_DR3_{gid}"
    nested_dir.mkdir()
    nested = nested_dir / f"Gaia_DR3_{gid}_summary.txt"
    lines = ["### STAR SUMMARY ###\n\n[PIPELINE RESULTS]\n"]
    for i in range(12):
        lines.append(
            f"Gaia_DR3_{gid}_epoch_{i + 1}.txt 6053{i}.5 -1.{i} 0.01 0.01 False\n"
        )
    nested.write_text("".join(lines))

    chosen = discover_summary_path(tmp_path, gid)
    assert chosen is not None
    assert chosen.resolve() == nested.resolve()
