from darkhunter_rv.website_table_csv import (
    clear_media_cells,
    clear_stray_plot_html,
    normalize_data_csv,
)


def test_normalize_data_csv_reorders_mass_and_clears_stray_img() -> None:
    hdr = ["GAIA NAME", "M2 (Msun)", "RV PLOT", "M2sin i (Msun)", "RV FIT", "FLUX PLOT"]
    data = ["id1", "1.0", "<img rv>", "2.0", "<img fit>", "<img hbeta>"]
    rows = [data]
    _, n_stray = normalize_data_csv(hdr, rows)
    assert hdr.index("M2sin i (Msun)") == hdr.index("M2 (Msun)") + 1
    assert rows[0][hdr.index("M2sin i (Msun)")] == "2.0"
    assert rows[0][hdr.index("RV PLOT")] == ""
    assert data[hdr.index("RV PLOT")] == ""
    assert n_stray >= 0


def test_clear_stray_plot_html_in_mass_column() -> None:
    hdr = ["M2 (Msun)", "M2sin i (Msun)", "RV PLOT"]
    data = ["1.0", '<a href="x"><img src="rv.png"></a>', ""]
    clear_stray_plot_html(hdr, [data])
    assert data[1] == ""
    assert data[0] == "1.0"


def test_clear_media_cells() -> None:
    hdr = ["RV PLOT", "M2 (Msun)"]
    data = ["<img>", "1.0"]
    clear_media_cells(hdr, [data])
    assert data[0] == ""
