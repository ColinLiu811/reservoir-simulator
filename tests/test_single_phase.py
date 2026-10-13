import numpy as np
import pytest

from ressim.grid import Grid
from ressim.single_phase import (
    SinglePhaseFluid,
    SinglePhaseSimulator,
    diffusivity,
    linear_step_solution,
)
from ressim.units import G, MILLIDARCY, CP

K = 100 * MILLIDARCY
MU = 1 * CP
CT = 1e-9
WATER = SinglePhaseFluid(mu=MU, ct=CT)


def test_steady_linear_flow_matches_darcy():
    g = Grid(nx=50, lx=500.0, ly=10.0, thickness=10.0, perm=K, poro=0.2)
    sim = SinglePhaseSimulator(g, WATER, 2e7, boundaries={"xmin": 2e7, "xmax": 1e7})
    sim.steady_state()
    assert np.allclose(sim.p, 2e7 - 1e7 * g.x / g.lx, rtol=1e-10)
    q_darcy = K * (g.ly * g.thickness) / MU * 1e7 / g.lx
    assert sim.boundary_flux("xmin") == pytest.approx(q_darcy, rel=1e-10)
    assert sim.boundary_flux("xmax") == pytest.approx(-q_darcy, rel=1e-10)


def test_layers_in_series_use_harmonic_average():
    k = np.where(np.arange(40) < 20, K, K / 10)
    g = Grid(nx=40, lx=400.0, ly=10.0, thickness=10.0, perm=k)
    sim = SinglePhaseSimulator(g, WATER, 0.0, boundaries={"xmin": 1e6, "xmax": 0.0})
    sim.steady_state()
    k_eff = 2.0 / (1 / K + 10 / K)
    q = k_eff * 100.0 / MU * 1e6 / 400.0
    assert sim.boundary_flux("xmin") == pytest.approx(q, rel=1e-10)


@pytest.mark.parametrize("solver", ["direct", "cg", "gmres"])
def test_transient_matches_erf_solution(solver):
    L, n = 2000.0, 400
    g = Grid(nx=n, lx=L, ly=10.0, thickness=10.0, perm=K, poro=0.2)
    sim = SinglePhaseSimulator(g, WATER, 2e7, boundaries={"xmin": 1e7}, solver=solver)
    t_end = 2e4                              # front still far from x = L
    sim.run(t_end, dt=5.0, dt_growth=1.1, dt_max=100.0)
    eta = diffusivity(K, 0.2, MU, CT)
    exact = linear_step_solution(g.x, t_end, 2e7, 1e7, eta)
    err = np.abs(sim.p - exact).max() / 1e7
    assert err < 5e-3


def test_backward_euler_is_first_order_in_time():
    L, n = 1000.0, 400
    eta = diffusivity(K, 0.2, MU, CT)
    t_end = 4e3
    errors = []
    for dt in (400.0, 200.0, 100.0):
        g = Grid(nx=n, lx=L, ly=1.0, perm=K, poro=0.2)
        sim = SinglePhaseSimulator(g, WATER, 1.0, boundaries={"xmin": 0.0})
        sim.run(t_end, dt=dt)
        errors.append(np.abs(sim.p - linear_step_solution(g.x, t_end, 1.0, 0.0, eta)).max())
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 0.8)


def test_material_balance_closes_each_step():
    g = Grid(nx=30, ny=20, lx=300.0, ly=200.0, thickness=5.0, perm=K)
    sim = SinglePhaseSimulator(g, WATER, 2e7, boundaries={"xmin": 2.5e7, "ymax": 1.5e7})
    for dt in (10.0, 100.0, 1000.0):
        p_old = sim.p.copy()
        sim.step(dt)
        assert sim.material_balance_error(p_old, dt) < 1e-9


def test_hydrostatic_column_is_in_equilibrium():
    g = Grid(nx=1, ny=20, lx=10.0, ly=100.0, thickness=10.0, perm=K, vertical=True, top=1500.0)
    p0 = 1e7 + WATER.rho * G * (g.depth - 1500.0)
    sim = SinglePhaseSimulator(g, WATER, p0, gravity=True)
    sim.step(1e4)
    assert np.allclose(sim.p, p0, rtol=1e-12)


def test_unknown_face_rejected():
    with pytest.raises(ValueError):
        SinglePhaseSimulator(Grid(nx=3), WATER, 0.0, boundaries={"left": 1.0})
