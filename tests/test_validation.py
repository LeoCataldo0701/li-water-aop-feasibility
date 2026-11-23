import pytest
from photofenton import simulate, load_parameters

# Tang et al. 2012, Chem. Eng. J. 184, 156-162, DOI 10.1016/j.cej.2012.01.020
# Conditions: 20 uM PFOA, pH 3.0, 30 mM H2O2, 2.0 mM Fe2+, 254 nm UV, 1 h
PFOA_NG_L = 20e-6 * 414.07 * 1e9
ARGS = ("PFOA", PFOA_NG_L, 3.0, 25.0, 30.0, 2000.0)


def test_h2o2_consumed_within_1h_matches_tang_2012():
    r = simulate(*ARGS, t_end=3600)
    assert r["H2O2_mM_t"][-1] < 0.1 * 30.0


def test_dark_fenton_negligible_matches_tang_2012():
    dark = {**load_parameters(), "k_hv_h2o2": 0.0, "k_hv_fe": 0.0}
    r = simulate(*ARGS, t_end=7200, params=dark)
    assert r["removal_final"] < 0.01


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: Tang 2012 reports 87.9% PFOA removal at 1 h; "
           "model gives ~6%. UV-activated PFOA chemistry and Fe3+-complex "
           "pathway are not modeled.",
)
def test_pfoa_removal_1h_matches_tang_2012():
    r = simulate(*ARGS, t_end=3600)
    assert r["removal_final"] > 0.8 * 0.879