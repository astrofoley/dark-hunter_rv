from darkhunter_rv.templates import _parse_lte_hires_filename


def test_parse_lte_solar_metallicity_suffix():
    fn = "lte12000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    assert _parse_lte_hires_filename(fn) == (12000.0, 4.0, 0.0)


def test_parse_lte_glued_mh_sign():
    fn = "lte11800-3.50+0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    assert _parse_lte_hires_filename(fn) == (11800.0, 3.5, 0.5)
