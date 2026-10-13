"""Validation studies: numerical results against analytical solutions.

Each function returns plain dicts/arrays so the CLI can print them and
:mod:`ressim.plots` can draw them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .buckley_leverett import BuckleyLeverett
from .grid import Grid, lognormal_perm
from .impes import IMPESOptions, IMPESSimulator
from .linalg import METHODS, assemble_laplacian, solve
from .relperm import CoreyRelPerm, TwoPhaseFluid, max_dfw_dsw, mobilities
from .single_phase import SinglePhaseFluid, SinglePhaseSimulator, diffusivity, linear_step_solution
from .units import CP, DAY, MILLIDARCY
from .wells import Well, face_wi, line_source_drawdown


@dataclass
class WaterfloodCase:
    """Reference 1D waterflood: 1000 ft x 100 ft x 30 ft, 100 md, 20% porosity,
    1 cp water displacing 5 cp oil, Corey exponents 2/2."""

    length: float = 304.8
    width: float = 30.48
    thickness: float = 9.144
    perm: float = 100 * MILLIDARCY
    poro: float = 0.2
    days_per_pv: float = 500.0
    p_out: float = 2e7
    relperm: CoreyRelPerm = None
    fluid: TwoPhaseFluid = None

    def __post_init__(self):
        if self.relperm is None:
            self.relperm = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8, nw=2, no=2)
        if self.fluid is None:
            self.fluid = TwoPhaseFluid(mu_w=1 * CP, mu_o=5 * CP)

    def grid(self, nx):
        return Grid(nx=nx, lx=self.length, ly=self.width, thickness=self.thickness,
                    perm=self.perm, poro=self.poro)

    def rate(self, grid):
        return float(np.sum(grid.pore_volume)) / (self.days_per_pv * DAY)

    def simulator(self, nx, options=None, **kw):
        g = self.grid(nx)
        wells = [Well("INJ", 0, "rate", self.rate(g)),
                 Well("PROD", nx - 1, "bhp", self.p_out, wi=face_wi(g, "xmax"))]
        sim = IMPESSimulator(g, self.fluid, self.relperm, self.relperm.swc, wells,
                             options, p_init=self.p_out, **kw)
        return g, sim

    def analytic(self):
        return BuckleyLeverett(self.relperm, self.fluid)


def l1_error(numeric, exact):
    return float(np.mean(np.abs(np.asarray(numeric) - np.asarray(exact))))


# --------------------------------------------------------------------------- 1
def buckley_leverett_comparison(nx=200, pvis=(0.1, 0.2, 0.3, 0.4), pvi_end=2.0,
                                case: WaterfloodCase | None = None):
    """Saturation profiles and production curves, IMPES vs Buckley-Leverett."""
    case = case or WaterfloodCase()
    bl = case.analytic()
    g, sim = case.simulator(nx)
    res = sim.run(pvi_end=pvi_end, report_pvi=pvis)
    xd = g.x / g.lx
    profiles = {}
    for v in pvis:
        sw = res.snapshots[("pvi", v)][1]
        exact = bl.profile(xd, v)
        profiles[v] = {"numeric": sw, "exact": exact, "l1": l1_error(sw, exact)}
    wc_exact, rf_exact = bl.performance(res.pvi)
    bt_num = float(res.pvi[np.argmax(res.water_cut > 0.01)])
    return {
        "nx": nx, "x_d": xd, "profiles": profiles, "result": res, "analytic": bl,
        "pvi": res.pvi, "water_cut": res.water_cut, "water_cut_exact": wc_exact,
        "recovery": res.recovery_factor, "recovery_exact": rf_exact,
        "breakthrough_numeric": bt_num, "breakthrough_exact": bl.breakthrough_pvi,
        "recovery_error_max": float(np.max(np.abs(res.recovery_factor - rf_exact))),
        "steps": res.n_steps, "mass_balance": res.max_mass_balance_error,
    }


# --------------------------------------------------------------------------- 2
def grid_convergence(nxs=(25, 50, 100, 200, 400, 800), pvi=0.3, case=None):
    """L1 saturation error vs grid size at fixed PVI.

    First-order upwind converges at O(dx) for smooth solutions but only about
    O(dx^0.5 to 1) here, because the Buckley-Leverett solution has a shock.
    """
    case = case or WaterfloodCase()
    bl = case.analytic()
    rows = []
    for nx in nxs:
        g, sim = case.simulator(nx)
        t0 = time.perf_counter()
        res = sim.run(pvi_end=pvi, report_pvi=(pvi,))
        sw = res.snapshots[("pvi", pvi)][1]
        rows.append({"nx": nx, "dx": g.dx, "l1": l1_error(sw, bl.profile(g.x / g.lx, pvi)),
                     "steps": res.n_steps, "seconds": time.perf_counter() - t0})
    dx = np.array([r["dx"] for r in rows])
    err = np.array([r["l1"] for r in rows])
    order = float(np.polyfit(np.log(dx), np.log(err), 1)[0])
    return {"rows": rows, "order": order, "pvi": pvi}


# --------------------------------------------------------------------------- 3
def cfl_study(nx=100, cfls=(0.5, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0), pvi=0.3, case=None):
    """Run with fixed steps at multiples of the CFL limit dx / max f'.

    The bound uses the largest f' over the whole saturation range, but the
    entropy solution jumps over the steepest part of f (it sits inside the
    shock), so the speeds actually present are at most the shock speed.
    The scheme therefore stays monotone a bit past CFL = 1, up to about
    max f' / shock speed, and loses accuracy and then boundedness beyond.
    """
    case = case or WaterfloodCase()
    bl = case.analytic()
    g = case.grid(nx)
    dt_cfl = g.pore_volume[0] / (case.rate(g) * max_dfw_dsw(case.relperm, case.fluid))
    rows = []
    for c in cfls:
        _, sim = case.simulator(nx, IMPESOptions(dt_fixed=c * dt_cfl, max_steps=200_000))
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                res = sim.run(pvi_end=pvi, report_pvi=(pvi,))
                sw = res.snapshots.get(("pvi", pvi), (None, sim.sw))[1]
                ok = np.all(np.isfinite(sw))
            except (FloatingPointError, RuntimeError, ValueError, ArithmeticError):
                sw, ok, res = sim.sw, False, None
        exact = bl.profile(g.x / g.lx, pvi)
        violation = sim.bound_violation() if ok else np.inf
        tv = float(np.sum(np.abs(np.diff(sw)))) if ok else np.inf
        rows.append({
            "cfl": c, "dt_days": c * dt_cfl / DAY, "stable": bool(ok and violation < 1e-9),
            # monotone: no new extrema, total variation no larger than the exact profile's
            "monotone": bool(ok and tv <= np.sum(np.abs(np.diff(exact))) + 1e-6),
            "violation": float(violation),
            "l1": l1_error(sw, exact) if ok else np.inf,
            "total_variation": tv,
            "sw": np.clip(sw, -1, 2) if ok else None,
        })
    fmax = max_dfw_dsw(case.relperm, case.fluid)
    return {"rows": rows, "x_d": g.x / g.lx, "exact": bl.profile(g.x / g.lx, pvi),
            "pvi": pvi, "dt_cfl_days": dt_cfl / DAY, "fprime_max": fmax,
            "shock_speed": bl.shock_speed, "effective_limit": fmax / bl.shock_speed}


# --------------------------------------------------------------------------- 4
def single_phase_checks():
    """Transient linear flow vs erf and radial drawdown vs line source."""
    k, mu, ct, phi = 100 * MILLIDARCY, 1 * CP, 1e-9, 0.2
    fluid = SinglePhaseFluid(mu=mu, ct=ct)
    out = {}

    g = Grid(nx=400, lx=2000.0, ly=10.0, thickness=10.0, perm=k, poro=phi)
    sim = SinglePhaseSimulator(g, fluid, 2e7, boundaries={"xmin": 1e7})
    t_end = 2e4
    sim.run(t_end, dt=5.0, dt_growth=1.1, dt_max=100.0)
    exact = linear_step_solution(g.x, t_end, 2e7, 1e7, diffusivity(k, phi, mu, ct))
    out["linear"] = {"x": g.x, "numeric": sim.p.copy(), "exact": exact, "t": t_end,
                     "max_rel_error": float(np.abs(sim.p - exact).max() / 1e7)}

    n, size, h, q = 101, 2000.0, 10.0, 1e-3
    g = Grid(nx=n, ny=n, lx=size, ly=size, thickness=h, perm=k, poro=phi)
    well = Well("P1", g.cell_at(size / 2, size / 2), "rate", -q, rw=0.1)
    sim = SinglePhaseSimulator(g, fluid, 2e7, wells=[well])
    res = sim.run(t_end, dt=10.0, dt_growth=1.15, dt_max=500.0)
    X, Y = np.meshgrid(g.x, g.y)
    r = np.hypot(X - size / 2, Y - size / 2).ravel()
    exact = 2e7 - line_source_drawdown(np.maximum(r, 1e-3), t_end, q, mu, k, h, phi, ct)
    ring = (r > 60) & (r < 300)
    pwf_exact = 2e7 - line_source_drawdown(well.rw, t_end, q, mu, k, h, phi, ct)
    out["radial"] = {
        "r": r, "numeric": sim.p.copy(), "exact": exact, "t": t_end,
        "max_rel_error": float(np.abs(sim.p[ring] - exact[ring]).max() / (2e7 - exact.min())),
        "bhp_numeric": float(res.well_bhp["P1"][-1]), "bhp_exact": float(pwf_exact),
        "bhp_times": res.times, "bhp_series": res.well_bhp["P1"],
        "bhp_series_exact": np.r_[2e7, 2e7 - line_source_drawdown(well.rw, res.times[1:], q, mu,
                                                                  k, h, phi, ct)],
    }
    return out


# --------------------------------------------------------------------------- 5
def five_spot(n=40, size=300.0, heterogeneous=True, pvi_end=1.0, seed=7, report_pvi=(0.2, 0.4, 0.7),
              solver="direct", case=None):
    """Quarter five-spot waterflood: injector and producer in opposite corners."""
    case = case or WaterfloodCase()
    perm = (lognormal_perm(n, n, case.perm, 1.0, corr_length=size / 10, seed=seed,
                           dx=size / n, dy=size / n) if heterogeneous else case.perm)
    g = Grid(nx=n, ny=n, lx=size, ly=size, thickness=case.thickness, perm=perm, poro=case.poro)
    q = float(np.sum(g.pore_volume)) / (case.days_per_pv * DAY)
    wells = [Well("INJ", 0, "rate", q, rw=0.1), Well("PROD", g.n - 1, "bhp", case.p_out, rw=0.1)]
    sim = IMPESSimulator(g, case.fluid, case.relperm, case.relperm.swc, wells,
                         IMPESOptions(solver=solver), p_init=case.p_out)
    t0 = time.perf_counter()
    res = sim.run(pvi_end=pvi_end, report_pvi=report_pvi)
    return {"grid": g, "result": res, "perm": perm, "seconds": time.perf_counter() - t0,
            "breakthrough": float(res.pvi[np.argmax(res.water_cut > 0.01)])}


# --------------------------------------------------------------------------- 6
def solver_benchmark(sizes=(50, 100, 200), methods=METHODS, tol=1e-8, seed=3):
    """Time each linear solver on a heterogeneous five-spot pressure matrix."""
    case = WaterfloodCase()
    rows = []
    for n in sizes:
        perm = lognormal_perm(n, n, case.perm, 1.5, corr_length=30.0, seed=seed,
                              dx=300.0 / n, dy=300.0 / n)
        g = Grid(nx=n, ny=n, lx=300.0, ly=300.0, thickness=case.thickness, perm=perm)
        c = g.connections()
        sw = np.full(g.n, case.relperm.swc)
        sw[: g.n // 3] = 0.6                          # partly flooded state
        lw, lo = mobilities(sw, case.relperm, case.fluid)
        lt = lw + lo
        coef = c.T * np.where(sw[c.a] >= sw[c.b], lt[c.a], lt[c.b])
        diag = np.zeros(g.n)
        wi = 2 * np.pi * case.perm * case.thickness / np.log(0.2 * g.dx / 0.1)
        diag[-1] = wi * lt[-1]
        rhs = np.zeros(g.n)
        rhs[0] = 1e-3
        rhs[-1] = diag[-1] * 2e7
        A = assemble_laplacian(g.n, c.a, c.b, coef, diag)
        ref = None
        for m in methods:
            if m == "cg-ssor" and n > 100:
                continue                              # python-level triangular solves: slow
            x, info = solve(A, rhs, method=m, tol=tol)
            ref = x if ref is None else ref
            rows.append({"cells": g.n, "method": m, "iterations": info.iterations,
                         "seconds": info.seconds, "residual": info.residual,
                         "max_dp_vs_first": float(np.abs(x - ref).max())})
    return rows
