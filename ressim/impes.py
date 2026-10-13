"""Two-phase incompressible water/oil flow with IMPES.

IMPES = IMplicit Pressure, Explicit Saturation. Each step:

1. Freeze saturations and solve the elliptic total-flux pressure equation

       -div( k * lam_t(S) * grad p ) = q_t

   implicitly with upstream-weighted face mobilities (one sparse solve).
2. Compute phase fluxes from the new pressure and advance water saturation
   with an explicit, first-order upwind finite-volume update

       phi V (S^{n+1} - S^n) / dt = q_w - sum_faces F_w

The explicit step is only conditionally stable. For the hyperbolic
Buckley-Leverett part the CFL restriction is

    dt <= phi V / (Q_out * max f'(S))

per cell, where ``Q_out`` is the total volumetric outflow of the cell. The
simulator chooses ``dt`` from this bound times a safety factor ``cfl``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .linalg import assemble_laplacian, solve
from .relperm import CoreyRelPerm, TwoPhaseFluid, max_dfw_dsw, mobilities
from .wells import Well


@dataclass
class IMPESOptions:
    cfl: float = 0.9               # safety factor on the CFL time step
    dt_fixed: float | None = None  # bypass the CFL control (stability studies)
    dt_max: float | None = None
    solver: str = "direct"
    max_steps: int = 2_000_000


@dataclass
class IMPESResult:
    times: np.ndarray
    pvi: np.ndarray
    dt: np.ndarray
    water_cut: np.ndarray            # producers combined, reservoir conditions
    oil_rate: np.ndarray             # m^3/s, positive
    water_rate: np.ndarray           # m^3/s produced, positive
    cum_oil: np.ndarray              # m^3
    recovery_factor: np.ndarray      # cumulative oil / initial oil in place
    snapshots: dict = field(default_factory=dict)  # report time -> (p, sw)
    n_steps: int = 0
    linear_iterations: int = 0
    max_mass_balance_error: float = 0.0
    max_bound_violation: float = 0.0


class IMPESSimulator:
    """Two-phase IMPES simulator on a :class:`~ressim.grid.Grid`.

    The primary unknowns are oil pressure ``p`` and water saturation ``sw``.
    Wells: a rate well with ``value > 0`` injects water; with ``value < 0`` it
    produces at fixed total rate split by the block fractional flow. BHP
    wells inject water if ``p_bhp > p_block`` and produce otherwise. At least
    one BHP well is needed to fix the pressure level of an incompressible
    system.
    """

    def __init__(self, grid: Grid, fluid: TwoPhaseFluid, relperm: CoreyRelPerm,
                 sw_init, wells: list[Well], options: IMPESOptions | None = None,
                 p_init=0.0):
        self.grid = grid
        self.fluid = fluid
        self.rp = relperm
        self.opts = options or IMPESOptions()
        self.wells = list(wells)
        if not any(w.control == "bhp" for w in self.wells):
            raise ValueError("incompressible IMPES needs at least one BHP-controlled well")
        self._wi = {w.name: w.index(grid) for w in self.wells}

        self.sw = np.broadcast_to(np.asarray(sw_init, dtype=float), (grid.n,)).copy()
        self.p = np.broadcast_to(np.asarray(p_init, dtype=float), (grid.n,)).copy()
        self.time = 0.0
        self.pv = grid.pore_volume
        self.ooip = float(np.sum(self.pv * (1.0 - self.sw)))
        self._water0 = float(np.sum(self.pv * self.sw))

        self.conn = grid.connections()
        self.fprime_max = max_dfw_dsw(relperm, fluid)

        self.cum_water_in = 0.0
        self.cum_water_out = 0.0
        self.cum_oil_out = 0.0
        self.linear_iterations = 0

    # ----------------------------------------------------------- mobilities
    def _upwind_mobilities(self, p):
        """Face mobilities taken from the upstream cell of each connection."""
        a, b = self.conn.a, self.conn.b
        lw, lo = mobilities(self.sw, self.rp, self.fluid)
        up = np.where(p[a] >= p[b], a, b)
        return lw[up], lo[up]

    # -------------------------------------------------------------- pressure
    def solve_pressure(self):
        """Implicit pressure solve with saturations frozen at time level n."""
        n = self.grid.n
        lw_f, lo_f = self._upwind_mobilities(self.p)
        coef = self.conn.T * (lw_f + lo_f)
        diag = np.zeros(n)
        rhs = np.zeros(n)
        lw, lo = mobilities(self.sw, self.rp, self.fluid)
        for w in self.wells:
            if w.control == "rate":
                rhs[w.cell] += w.value
            else:
                c = self._wi[w.name] * (lw[w.cell] + lo[w.cell])
                diag[w.cell] += c
                rhs[w.cell] += c * w.value
        A = assemble_laplacian(n, self.conn.a, self.conn.b, coef, diag)
        self.p, info = solve(A, rhs, method=self.opts.solver, x0=self.p)
        self.linear_iterations += info.iterations
        return info

    # ---------------------------------------------------------------- fluxes
    def phase_fluxes(self):
        """Water and oil volumetric fluxes [m^3/s] from ``a`` to ``b``."""
        p = self.p
        a, b = self.conn.a, self.conn.b
        lw_f, lo_f = self._upwind_mobilities(p)
        dp = p[a] - p[b]
        return self.conn.T * lw_f * dp, self.conn.T * lo_f * dp

    def well_rates(self):
        """Per-well (q_w, q_o) [m^3/s] at the current state, + = injection."""
        lw, lo = mobilities(self.sw, self.rp, self.fluid)
        out = {}
        for w in self.wells:
            c = w.cell
            lt = lw[c] + lo[c]
            if w.control == "rate":
                qt = w.value
            else:
                qt = self._wi[w.name] * lt * (w.value - self.p[c])
            if qt >= 0:
                out[w.name] = (qt, 0.0)
            else:
                fw = lw[c] / lt if lt > 0 else 0.0
                out[w.name] = (fw * qt, (1.0 - fw) * qt)
        return out

    # ------------------------------------------------------------- time step
    def stable_dt(self, fw_flux, fo_flux, rates):
        """CFL-limited step for the explicit saturation update."""
        ft = fw_flux + fo_flux
        q_out = np.zeros(self.grid.n)
        np.add.at(q_out, self.conn.a, np.maximum(ft, 0.0))
        np.add.at(q_out, self.conn.b, np.maximum(-ft, 0.0))
        for w in self.wells:
            qw, qo = rates[w.name]
            q_out[w.cell] += max(-(qw + qo), 0.0)
        with np.errstate(divide="ignore"):
            dt_cells = self.pv / (q_out * self.fprime_max)
        return self.opts.cfl * float(np.min(dt_cells))

    def step(self, dt_limit=None):
        """One IMPES step. Returns the step length taken [s]."""
        self.solve_pressure()
        fw_flux, fo_flux = self.phase_fluxes()
        rates = self.well_rates()

        if self.opts.dt_fixed is not None:
            dt = self.opts.dt_fixed
        else:
            dt = self.stable_dt(fw_flux, fo_flux, rates)
            if self.opts.dt_max is not None:
                dt = min(dt, self.opts.dt_max)
        if dt_limit is not None:
            dt = min(dt, dt_limit)
        if not np.isfinite(dt) or dt <= 0:
            raise RuntimeError("no flow in the model: cannot choose a time step")

        # explicit upwind water balance
        net_out = np.zeros(self.grid.n)
        np.add.at(net_out, self.conn.a, fw_flux)
        np.add.at(net_out, self.conn.b, -fw_flux)
        src = np.zeros(self.grid.n)
        for w in self.wells:
            src[w.cell] += rates[w.name][0]
        self.sw = self.sw + dt / self.pv * (src - net_out)

        for w in self.wells:
            qw, qo = rates[w.name]
            if qw > 0:
                self.cum_water_in += qw * dt
            else:
                self.cum_water_out -= qw * dt
            self.cum_oil_out -= qo * dt
        self.time += dt
        self._last_rates = rates
        return dt

    # ------------------------------------------------------------ diagnostics
    def mass_balance_error(self):
        """|change in water in place - (injected - produced)| / injected."""
        dw = float(np.sum(self.pv * self.sw)) - self._water0
        net = self.cum_water_in - self.cum_water_out
        return abs(dw - net) / max(self.cum_water_in, 1e-300)

    def bound_violation(self):
        """How far saturations stray outside [Swc, 1 - Sor] (0 if inside)."""
        lo = self.rp.swc - self.sw
        hi = self.sw - (1.0 - self.rp.sor)
        return float(max(lo.max(), hi.max(), 0.0))

    @property
    def pvi(self):
        return self.cum_water_in / float(np.sum(self.pv))

    # -------------------------------------------------------------------- run
    def run(self, t_end=None, pvi_end=None, report_pvi=(), report_times=()):
        """March until ``t_end`` [s] or ``pvi_end`` injected pore volumes.

        Steps are shortened to land exactly on ``report_times``; for
        ``report_pvi`` the snapshot is taken at the first step that reaches
        the requested PVI, with the step shortened using the current
        injection rate so the overshoot is negligible.
        """
        if (t_end is None) == (pvi_end is None):
            raise ValueError("give exactly one of t_end or pvi_end")
        report_times = sorted(report_times)
        report_pvi = sorted(report_pvi)
        hist = {k: [] for k in ("t", "pvi", "dt", "wc", "qo", "qw", "np")}
        snaps = {}
        total_pv = float(np.sum(self.pv))
        max_mb = 0.0
        max_violation = 0.0
        steps = 0

        def done():
            if t_end is not None:
                return self.time >= t_end * (1 - 1e-12)
            return self.pvi >= pvi_end * (1 - 1e-12)

        while not done():
            if steps >= self.opts.max_steps:
                warnings.warn("IMPES stopped at max_steps", RuntimeWarning)
                break
            limit = None
            targets_t = [t for t in report_times if t > self.time * (1 + 1e-12)]
            if t_end is not None:
                targets_t.append(t_end)
            if targets_t:
                limit = min(targets_t) - self.time
            targets_v = [v for v in report_pvi if v > self.pvi * (1 + 1e-12)]
            if pvi_end is not None:
                targets_v.append(pvi_end)
            if targets_v:
                q_in = self._injection_rate_estimate()
                if q_in > 0:
                    dt_v = (min(targets_v) - self.pvi) * total_pv / q_in
                    limit = dt_v if limit is None else min(limit, dt_v)

            dt = self.step(limit)
            steps += 1
            rates = self._last_rates
            qw = -sum(min(r[0], 0.0) for r in rates.values())
            qo = -sum(r[1] for r in rates.values())
            hist["t"].append(self.time)
            hist["pvi"].append(self.pvi)
            hist["dt"].append(dt)
            hist["wc"].append(qw / (qw + qo) if qw + qo > 0 else 0.0)
            hist["qo"].append(qo)
            hist["qw"].append(qw)
            hist["np"].append(self.cum_oil_out)
            max_mb = max(max_mb, self.mass_balance_error())
            max_violation = max(max_violation, self.bound_violation())

            for t in report_times:
                if t not in snaps and self.time >= t * (1 - 1e-9):
                    snaps[t] = (self.p.copy(), self.sw.copy())
            for v in report_pvi:
                if ("pvi", v) not in snaps and self.pvi >= v * (1 - 1e-6):
                    snaps[("pvi", v)] = (self.p.copy(), self.sw.copy())

        a = {k: np.array(v) for k, v in hist.items()}
        return IMPESResult(
            times=a["t"], pvi=a["pvi"], dt=a["dt"], water_cut=a["wc"],
            oil_rate=a["qo"], water_rate=a["qw"], cum_oil=a["np"],
            recovery_factor=a["np"] / self.ooip, snapshots=snaps, n_steps=steps,
            linear_iterations=self.linear_iterations,
            max_mass_balance_error=max_mb, max_bound_violation=max_violation,
        )

    def _injection_rate_estimate(self):
        rates = getattr(self, "_last_rates", None)
        if rates is None:
            return sum(w.value for w in self.wells if w.control == "rate" and w.value > 0)
        return sum(max(r[0], 0.0) for r in rates.values())
