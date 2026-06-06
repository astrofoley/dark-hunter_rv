from darkhunter_rv.website_table_csv import LEGACY_NEXT_RV_MJD, normalize_data_csv


def test_normalize_drops_legacy_next_rv_mjd_column() -> None:
    hdr = [
        "GAIA NAME",
        "M2 (Msun)",
        "NEXT RV EVENT (DATE)",
        LEGACY_NEXT_RV_MJD,
    ]
    rows = [["Gaia DR3 123", "0.5", "2024/01/01", "60000"]]
    normalize_data_csv(hdr, rows)
    assert LEGACY_NEXT_RV_MJD not in hdr
    assert "NEXT RV EVENT (DATE)" in hdr
    assert rows[0][hdr.index("NEXT RV EVENT (DATE)")] == "2024/01/01"


def test_normalize_adds_inclination_column() -> None:
    hdr = ["GAIA NAME", "M2 (Msun)", "M2sin i (Msun)", "(M2sin i)/(sin i) (Msun)"]
    rows = [["id", "1", "0.2", "0.3"]]
    normalize_data_csv(hdr, rows)
    assert "INCLINATION (deg)" in hdr
