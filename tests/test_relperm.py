import numpy as np
import pytest

from ressim.relperm import (
    CapillaryPressure,
    CoreyRelPerm,
    TwoPhaseFluid,
    fractional_flow,
    fractional_flow_derivative,
    max_dfw_dsw,
    mobility_ratio,
)

RP = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8, nw=2, no=2)
FL = TwoPhaseFluid(mu_w=1e-3, mu_o=5e-3)


def test_endpoints():
    assert RP.krw(0.2) == 0 and RP.kro(0.2) == pytest.approx(0.8)
    assert RP.krw(0.8) == pytest.approx(0.3) and RP.kro(0.8) == 0
    # clipped outside the mobile range
    assert RP.krw(0.1) == 0 and RP.kro(0.95) == 0


def test_fractional_flow_is_monotone_s_curve():
    s = np.linspace(0.2, 0.8, 200)
    f = fractional_flow(s, RP, FL)
    assert f[0] == 0 and f[-1] == pytest.approx(1)
    assert np.all(np.diff(f) >= 0)
    d2 = np.diff(f, 2)
    assert d2[:20].min() > 0 and d2[-20:].max() < 0     # convex then concave


def test_derivative_against_analytic():
    s = 0.5
    se = RP.se(s)
    lw = 0.3 * se**2 / 1e-3
    lo = 0.8 * (1 - se) ** 2 / 5e-3
    dlw = 0.3 * 2 * se / 1e-3 / 0.6
    dlo = -0.8 * 2 * (1 - se) / 5e-3 / 0.6
    expect = (dlw * lo - lw * dlo) / (lw + lo) ** 2
    assert fractional_flow_derivative(s, RP, FL) == pytest.approx(expect, rel=1e-6)
    assert max_dfw_dsw(RP, FL) >= expect


def test_mobility_ratio():
    assert mobility_ratio(RP, FL) == pytest.approx((0.3 / 1e-3) / (0.8 / 5e-3))


def test_capillary_pressure():
    pc = CapillaryPressure(pe=1e4, lam=2.0, swc=0.2, sor=0.2)
    assert pc(0.8) == pytest.approx(1e4)
    assert pc(0.5) > pc(0.6) > pc(0.8)
    assert pc.derivative(0.5) < 0
    assert np.all(CapillaryPressure()(np.array([0.3, 0.5])) == 0)


def test_invalid_endpoints():
    with pytest.raises(ValueError):
        CoreyRelPerm(swc=0.6, sor=0.5)
