from darkhunter_rv.rv_point_filters import rv_value_is_valid


def test_rejects_legacy_sentinel() -> None:
    assert not rv_value_is_valid(-9999.0)
    assert not rv_value_is_valid(-10000.0)


def test_accepts_normal_rv() -> None:
    assert rv_value_is_valid(-15.5)
    assert rv_value_is_valid(120.0)
