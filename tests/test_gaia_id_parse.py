"""Gaia DR3 source_id extraction from paths (16–19 digit ids)."""

from darkhunter_rv.gaia_utils import GAIA_DR3_ID_RE, parse_gaia_id, parse_gaia_id_from_path

STAR_17 = 77413727493690112
STAR_19 = 1702370142434513152


def test_gaia_dr3_id_re_17_digits():
    m = GAIA_DR3_ID_RE.search(f"Gaia_DR3_{STAR_17}_epoch_1.txt")
    assert m is not None
    assert int(m.group(1)) == STAR_17


def test_parse_gaia_id_from_path_17_digit_epoch():
    path = f"/data/Gaia_DR3_{STAR_17}_epoch_8.txt"
    assert parse_gaia_id_from_path(path) == STAR_17
    assert parse_gaia_id(path) == STAR_17


def test_parse_gaia_id_from_path_19_digit_epoch():
    path = f"Gaia_DR3_{STAR_19}_epoch_1.txt"
    assert parse_gaia_id_from_path(path) == STAR_19


def test_parse_gaia_id_rejects_too_short():
    assert parse_gaia_id_from_path("Gaia_DR3_12345_epoch_1.txt") is None
