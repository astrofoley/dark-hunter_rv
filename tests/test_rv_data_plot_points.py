from darkhunter_rv.rv_keplerian_plots import points_for_rv_data_plot


class _Pt:
    def __init__(self, *, literature: bool) -> None:
        self.is_literature = literature
        self.mjd = 60000.0
        self.rv = 1.0
        self.rv_err = 0.1
        self.telescope = "LAMOST"


def test_points_for_rv_data_plot_prefers_our_epochs() -> None:
    ours = _Pt(literature=False)
    ours.telescope = "APF"
    lit = _Pt(literature=True)
    pts, lit_only = points_for_rv_data_plot([lit, ours])
    assert len(pts) == 1
    assert pts[0].telescope == "APF"
    assert lit_only is False


def test_points_for_rv_data_plot_falls_back_to_literature() -> None:
    lit = _Pt(literature=True)
    pts, lit_only = points_for_rv_data_plot([lit])
    assert len(pts) == 1
    assert lit_only is True
