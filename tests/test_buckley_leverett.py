import numpy as np
import pytest

from ressim.buckley_leverett import BuckleyLeverett
from ressim.relperm import CoreyRelPerm, TwoPhaseFluid, fractional_flow_derivative

RP = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8, nw=2, no=2)
FL = TwoPhaseFluid(mu_w=1e-3, mu_o=5e-3)


@pytest.fixture
def bl():
    return BuckleyLeverett(RP, FL)


def test_welge_tangency(bl):
    sf = bl.sw_front
    secant = (bl.f(sf) - bl.f_i) / (sf - bl.swi)
    assert secant == pytest.approx(fractional_flow_derivative(sf, RP, FL), rel=1e-5)
    assert secant == pytest.approx(bl.shock_speed, rel=1e-10)


def test_profile_conserves_water(bl):
    x = np.linspace(0, 1, 200001)
    for pvi in (0.1, 0.25, 0.4):
        s = bl.profile(x, pvi)
        injected = np.trapezoid(s - bl.swi, x)
        assert injected == pytest.approx(pvi, rel=2e-3)


def test_profile_shape(bl):
    s = bl.profile(np.array([0.0, 0.1, 0.25, 0.9]), 0.3)
    assert s[0] == pytest.approx(0.8, abs=1e-3)
    assert s[0] > s[1] > bl.sw_front > s[3] == bl.swi


def test_recovery_continuous_at_breakthrough(bl):
    tb = bl.breakthrough_pvi
    before, after = bl.outlet(tb * (1 - 1e-6)), bl.outlet(tb * (1 + 1e-6))
    assert before.recovery_factor == pytest.approx(after.recovery_factor, rel=1e-4)
    assert before.water_cut == 0 and after.water_cut == pytest.approx(bl.f(bl.sw_front), rel=1e-3)


def test_recovery_approaches_movable_oil(bl):
    late = bl.outlet(1000.0)
    assert late.recovery_factor == pytest.approx((1 - 0.2 - 0.2) / (1 - 0.2), abs=5e-3)
    wc, rf = bl.performance([0.5, 1.0, 2.0])
    assert np.all(np.diff(wc) > 0) and np.all(np.diff(rf) > 0)


def test_unit_mobility_linear_curves_give_piston_shock():
    rp = CoreyRelPerm(swc=0.1, sor=0.3, krw0=1.0, kro0=1.0, nw=1, no=1)
    bl = BuckleyLeverett(rp, TwoPhaseFluid(1e-3, 1e-3))
    assert bl.sw_front == pytest.approx(0.7)
    assert bl.breakthrough_pvi == pytest.approx(0.6)
    s = bl.profile([0.4, 0.6], 0.3)          # front at x_D = 0.5
    assert s.tolist() == pytest.approx([0.7, 0.1])


def test_initial_mobile_water():
    bl = BuckleyLeverett(RP, FL, swi=0.3)
    assert bl.f_i > 0
    assert bl.sw_front > 0.3
    st = bl.outlet(0.1)
    assert st.water_cut == pytest.approx(bl.f_i)
    assert st.oil_recovered_pv == pytest.approx(0.1 * (1 - bl.f_i))


def test_bad_initial_saturation():
    with pytest.raises(ValueError):
        BuckleyLeverett(RP, FL, swi=0.85)
