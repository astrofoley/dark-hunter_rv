import numpy as np

from darkhunter_rv import qc


def test_fft_ccf_rss_ratio_separates_noise_from_bump():
    x = np.linspace(-1000, 1000, 500)
    rng = np.random.default_rng(42)
    y_noise = 1200.0 + rng.normal(0, 12.0, size=x.shape)
    _, _, m_n = qc.fft_ccf_passes_vs_flat(x, y_noise, max_rss_ratio=0.99)
    rng2 = np.random.default_rng(7)
    y_bump = (
        900.0
        + 950.0 * np.exp(-0.5 * ((x - 25.0) / 75.0) ** 2)
        + rng2.normal(0, 9.0, size=x.shape)
    )
    ok_b, reason_b, m_b = qc.fft_ccf_passes_vs_flat(x, y_bump, max_rss_ratio=0.88)
    assert np.isfinite(m_n["fft_ccf_rss_ratio"]) and np.isfinite(m_b["fft_ccf_rss_ratio"])
    assert m_n["fft_ccf_rss_ratio"] > m_b["fft_ccf_rss_ratio"] + 0.15
    assert ok_b and reason_b == "ok"
    assert m_b["fft_ccf_rss_ratio"] < 0.65
