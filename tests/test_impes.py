import numpy as np
import pytest

from ressim.buckley_leverett import BuckleyLeverett
from ressim.grid import Grid
from ressim.impes import IMPESOptions, IMPESSimulator
from ressim.relperm import CoreyRelPerm, TwoPhaseFluid
from ressim.wells import Well, face_wi

RP = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8, nw=2, no=2)
FL = TwoPhaseFluid(mu_w=1e-3, mu_o=5e-3)


def linear_flood(nx, options=None, perm=1e-13, **kw):
    g = Grid(nx=nx, lx=300.0, ly=30.0, thickness=10.0, perm=perm, poro=0.2)
    q = float(np.sum(g.pore_volume)) / (200 * 86400.0)      # 1 PV in 200 days
    wells = [Well("I", 0, "rate", q),
             Well("P", nx - 1, "bhp", 2e7, wi=face_wi(g, "xmax"))]
    return g, IMPESSimulator(g, FL, RP, RP.swc, wells, options, p_init=2e7, **kw)


def test_profile_matches_buckley_leverett():
    g, sim = linear_flood(200)
    res = sim.run(pvi_end=0.3, report_pvi=(0.3,))
    _, sw = res.snapshots[("pvi", 0.3)]
    exact = BuckleyLeverett(RP, FL).profile(g.x / g.lx, 0.3)
    assert np.mean(np.abs(sw - exact)) < 0.006


def test_breakthrough_and_recovery_match_analytic():
    _, sim = linear_flood(200)
    res = sim.run(pvi_end=1.5)
    bl = BuckleyLeverett(RP, FL)
    bt = res.pvi[np.argmax(res.water_cut > 0.01)]
    assert bt == pytest.approx(bl.breakthrough_pvi, rel=0.03)
    rf_exact = bl.performance(res.pvi[::50])[1]
    assert np.abs(res.recovery_factor[::50] - rf_exact).max() < 0.01


def test_water_is_conserved_and_saturations_bounded():
    _, sim = linear_flood(100)
    res = sim.run(pvi_end=1.0)
    assert res.max_mass_balance_error < 1e-12
    assert res.max_bound_violation < 1e-12
    assert res.pvi[-1] == pytest.approx(1.0, rel=1e-9)


def test_incompressible_rates_balance():
    _, sim = linear_flood(50)
    sim.run(pvi_end=0.6)
    sim.solve_pressure()             # rates are consistent with a fresh pressure
    rates = sim.well_rates()
    total = sum(qw + qo for qw, qo in rates.values())
    assert abs(total) < 1e-10 * rates["I"][0]


def test_quarter_five_spot_is_symmetric_about_diagonal():
    n = 21
    g = Grid(nx=n, ny=n, lx=200.0, ly=200.0, thickness=10.0, perm=1e-13)
    wells = [Well("I", 0, "rate", 1e-3), Well("P", g.n - 1, "bhp", 2e7)]
    sim = IMPESSimulator(g, FL, RP, RP.swc, wells, p_init=2e7)
    sim.run(pvi_end=0.3)
    s = g.reshape(sim.sw)
    assert np.allclose(s, s.T, atol=1e-10)


def test_needs_a_pressure_controlled_well():
    g = Grid(nx=5)
    with pytest.raises(ValueError):
        IMPESSimulator(g, FL, RP, 0.2, [Well("I", 0, "rate", 1.0)])


def test_reports_land_on_requested_times():
    _, sim = linear_flood(40)
    day = 86400.0
    res = sim.run(t_end=30 * day, report_times=(10 * day, 20 * day))
    assert set(res.snapshots) == {10 * day, 20 * day}
    assert res.times[-1] == pytest.approx(30 * day)
    assert np.any(np.isclose(res.times, 10 * day))


# ---------------------------------------------------------------- Pc and gravity
from ressim.relperm import CapillaryPressure  # noqa: E402

DAY = 86400.0


def _column(sw_top, sw_bottom, n=20):
    g = Grid(nx=1, ny=n, lx=10.0, ly=20.0, thickness=10.0, perm=5e-13, poro=0.25,
             vertical=True, top=2000.0)
    sw = np.where(np.arange(n) < n // 2, sw_top, sw_bottom)
    return g, sw


def test_stable_stratification_stays_in_equilibrium():
    # oil (light) above water (heavy): no flow once pressure is hydrostatic
    g, sw = _column(RP.swc, 1 - RP.sor)
    sim = IMPESSimulator(g, FL, RP, sw, options=IMPESOptions(dt_max=DAY), p_init=2e7,
                         gravity=True)
    sim.run(t_end=50 * DAY)
    assert np.allclose(sim.sw, sw, atol=1e-10)


def test_gravity_segregation_moves_water_down():
    g, sw = _column(1 - RP.sor, RP.swc)          # water on top: unstable
    sim = IMPESSimulator(g, FL, RP, sw, options=IMPESOptions(ds_max=0.05), p_init=2e7,
                         gravity=True)
    res = sim.run(t_end=2000 * DAY)
    top, bottom = sim.sw[: g.ny // 2].mean(), sim.sw[g.ny // 2:].mean()
    assert bottom > 0.7 and top < 0.3
    assert sim.sw.sum() == pytest.approx(sw.sum(), rel=1e-12)       # closed system
    assert res.max_bound_violation < 1e-6


def test_capillary_pressure_smears_the_front():
    # 1 darcy and a 1 bar entry pressure: capillary length D/u ~ several cells
    pc = CapillaryPressure(pe=1e5, lam=2.0, swc=RP.swc, sor=RP.sor)
    g, plain = linear_flood(100, perm=1e-12)
    _, capil = linear_flood(100, perm=1e-12, capillary=pc)
    s0 = plain.run(pvi_end=0.3, report_pvi=(0.3,)).snapshots[("pvi", 0.3)][1]
    res = capil.run(pvi_end=0.3, report_pvi=(0.3,))
    s1 = res.snapshots[("pvi", 0.3)][1]

    def width(s):            # length of the 0.25 < Sw < 0.5 transition zone
        return np.sum((s > 0.25) & (s < 0.5)) * g.dx

    assert width(s1) > 3 * width(s0)
    assert res.max_mass_balance_error < 1e-12
    assert res.max_bound_violation < 1e-9


def test_capillary_diffusion_limits_the_step():
    pc = CapillaryPressure(pe=3e5, lam=2.0, swc=RP.swc, sor=RP.sor)
    _, sim = linear_flood(200, perm=1e-12, capillary=pc)
    sim.sw[:40] = np.linspace(0.8, 0.3, 40)
    sim.solve_pressure()
    rates = sim.well_rates()
    dt_pc = sim.stable_dt(*sim.phase_fluxes(), rates)
    sim.pc = CapillaryPressure()
    dt_adv = sim.stable_dt(*sim.phase_fluxes(), rates)
    assert dt_pc < 0.5 * dt_adv
