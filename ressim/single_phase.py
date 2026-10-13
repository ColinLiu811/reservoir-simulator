"""Implicit single-phase slightly compressible flow.

Discretises

    phi * ct * dp/dt = div( k/mu * grad(p - rho*g*z) ) + q

with two-point fluxes in space and backward Euler in time. With constant
``ct`` and ``mu`` the system is linear, so each step is one sparse solve.
Backward Euler is unconditionally stable, which is why the pressure
equation is always solved implicitly (here and inside IMPES).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import FACES, Grid
from .linalg import SolveInfo, assemble_laplacian, solve
from .units import G


@dataclass
class SinglePhaseFluid:
    """Slightly compressible fluid.

    mu  : viscosity [Pa.s]
    ct  : total compressibility, fluid + pore volume [1/Pa]
    rho : density [kg/m^3], only used with gravity
    """

    mu: float
    ct: float
    rho: float = 1000.0


@dataclass
class SinglePhaseResult:
    times: np.ndarray
    pressure: np.ndarray                  # (n_times, n_cells)
    boundary_flux: dict = field(default_factory=dict)   # face -> (n_times,) inflow m^3/s
    solver_info: list = field(default_factory=list)


class SinglePhaseSimulator:
    """Backward-Euler single-phase simulator.

    Parameters
    ----------
    grid : :class:`~ressim.grid.Grid`
    fluid : :class:`SinglePhaseFluid`
    p_init : initial pressure, scalar or per-cell [Pa]
    boundaries : ``{face: pressure}`` for constant-pressure faces; faces not
        listed are no-flow
    gravity : include the ``rho*g*z`` term (needs a vertical grid)
    solver : linear solver method, see :mod:`ressim.linalg`
    """

    def __init__(self, grid: Grid, fluid: SinglePhaseFluid, p_init, boundaries=None,
                 gravity=False, solver="direct"):
        self.grid = grid
        self.fluid = fluid
        self.p = np.broadcast_to(np.asarray(p_init, dtype=float), (grid.n,)).copy()
        self.boundaries = dict(boundaries or {})
        for face in self.boundaries:
            if face not in FACES:
                raise ValueError(f"unknown face {face!r}")
        self.gravity = gravity
        self.solver = solver
        self.time = 0.0

        conn = grid.connections()
        self._conn = conn
        self._lap_coef = conn.T / fluid.mu
        self._grav_rhs = np.zeros(grid.n)
        if gravity:
            z = grid.depth
            flux = self._lap_coef * fluid.rho * G * (z[conn.a] - z[conn.b])
            np.add.at(self._grav_rhs, conn.a, flux)
            np.add.at(self._grav_rhs, conn.b, -flux)

    # ------------------------------------------------------------------
    def _assemble(self, dt):
        g, f = self.grid, self.fluid
        n = g.n
        diag = np.zeros(n)
        rhs = self._grav_rhs.copy()
        if dt is not None:
            acc = g.pore_volume * f.ct / dt
            diag += acc
            rhs += acc * self.p
        for face, pb in self.boundaries.items():
            cells, Tb = g.boundary_cells(face)
            np.add.at(diag, cells, Tb / f.mu)
            np.add.at(rhs, cells, Tb / f.mu * pb)
        self._well_terms(diag, rhs)
        A = assemble_laplacian(n, self._conn.a, self._conn.b, self._lap_coef, diag)
        return A, rhs

    def _well_terms(self, diag, rhs):
        """Hook for well source terms (none in the base model)."""

    def step(self, dt) -> SolveInfo:
        """Advance one backward-Euler step of length ``dt`` [s]."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        A, rhs = self._assemble(dt)
        self.p, info = solve(A, rhs, method=self.solver, x0=self.p)
        self.time += dt
        return info

    def steady_state(self) -> SolveInfo:
        """Solve the incompressible (dp/dt = 0) problem in place."""
        A, rhs = self._assemble(None)
        self.p, info = solve(A, rhs, method=self.solver)
        return info

    def boundary_flux(self, face) -> float:
        """Total volumetric inflow through a constant-pressure face [m^3/s]."""
        cells, Tb = self.grid.boundary_cells(face)
        pb = self.boundaries[face]
        return float(np.sum(Tb / self.fluid.mu * (pb - self.p[cells])))

    def run(self, t_end, dt, dt_growth=1.0, dt_max=None) -> SinglePhaseResult:
        """March to ``t_end``; optionally grow the step geometrically.

        Small early steps resolve the fast initial transient, growth keeps
        the step count low once the pressure field smooths out.
        """
        times, ps, infos = [self.time], [self.p.copy()], []
        flux = {f: [self.boundary_flux(f)] for f in self.boundaries}
        while self.time < t_end * (1 - 1e-12):
            step = min(dt, t_end - self.time)
            infos.append(self.step(step))
            times.append(self.time)
            ps.append(self.p.copy())
            for f in flux:
                flux[f].append(self.boundary_flux(f))
            dt *= dt_growth
            if dt_max is not None:
                dt = min(dt, dt_max)
        return SinglePhaseResult(
            times=np.array(times), pressure=np.array(ps),
            boundary_flux={f: np.array(v) for f, v in flux.items()}, solver_info=infos,
        )

    def material_balance_error(self, p_old, dt) -> float:
        """Relative mismatch between storage change and net inflow over the
        last step (call right after :meth:`step`)."""
        storage = np.sum(self.grid.pore_volume * self.fluid.ct * (self.p - p_old))
        inflow = (sum(self.boundary_flux(f) for f in self.boundaries) + self._well_inflow()) * dt
        scale = max(abs(storage), abs(inflow), 1e-300)
        return abs(storage - inflow) / scale

    def _well_inflow(self) -> float:
        return 0.0


def diffusivity(k, phi, mu, ct):
    """Hydraulic diffusivity eta = k / (phi mu ct) [m^2/s]."""
    return k / (phi * mu * ct)


def linear_step_solution(x, t, p_init, p_boundary, eta):
    """Pressure in a semi-infinite 1D medium after the face at x=0 is
    stepped from ``p_init`` to ``p_boundary`` at t=0:

        p = p_b + (p_i - p_b) * erf(x / (2 sqrt(eta t)))

    >>> float(linear_step_solution(0.0, 1.0, 10.0, 0.0, 1.0))
    0.0
    """
    from scipy.special import erf

    return p_boundary + (p_init - p_boundary) * erf(np.asarray(x) / (2.0 * np.sqrt(eta * t)))
