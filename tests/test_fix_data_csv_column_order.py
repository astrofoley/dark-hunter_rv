import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fix_data_csv_column_order",
    _ROOT / "scripts" / "fix_data_csv_column_order.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
move_columns_after = _mod.move_columns_after
clear_media_cells = _mod.clear_media_cells
clear_stray_plot_html = _mod.clear_stray_plot_html


def test_move_columns_after_reorders_row_values(tmp_path: Path) -> None:
    hdr = ["GAIA NAME", "M2 (Msun)", "RV PLOT", "M2sin i (Msun)", "RV FIT"]
    data = ["id1", "1.0", "<img rv>", "2.0", "<img fit>"]
    move_columns_after(hdr, [data], ["M2sin i (Msun)"], "M2 (Msun)")
    assert hdr == ["GAIA NAME", "M2 (Msun)", "M2sin i (Msun)", "RV PLOT", "RV FIT"]
    assert data == ["id1", "1.0", "2.0", "<img rv>", "<img fit>"]


def test_clear_stray_plot_html_in_mass_column() -> None:
    hdr = ["M2 (Msun)", "M2sin i (Msun)", "RV PLOT"]
    data = ["1.0", '<a href="x"><img src="rv.png"></a>', ""]
    clear_stray_plot_html(hdr, [data])
    assert data[1] == ""
    assert data[0] == "1.0"


def test_clear_media_cells(tmp_path: Path) -> None:
    hdr = ["RV PLOT", "M2 (Msun)"]
    data = ["<img>", "1.0"]
    clear_media_cells(hdr, [data])
    assert data[0] == ""
