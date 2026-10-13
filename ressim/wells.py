"""Vertical wells with Peaceman's well index.

A well couples the wellbore pressure ``p_bhp`` to its grid-block pressure:

    q = WI * lambda * (p_bhp - p_block)        (q > 0 = injection)

Peaceman (1978, 1983) showed that the block pressure of a five-point
finite-difference scheme equals the steady radial pressure at an
equivalent radius r0 ~ 0.2 dx, which gives

    WI = 2 pi sqrt(kx ky) h / (ln(r0 / rw) + skin)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


@dataclass
class Well:
    """Vertical well completed in a single cell.

    control : ``"rate"`` (``value`` = surface-free reservoir rate, m^3/s,
        > 0 injects) or ``"bhp"`` (``value`` = bottom-hole pressure, Pa)
    rw, skin : wellbore radius [m] and skin factor, used for Peaceman WI
    wi : explicit well index [m^3]; overrides Peaceman (useful in 1D, where
        a "well" really stands for an inflow/outflow face)
    """

    name: str
    cell: int
    control: str
    value: float
    rw: float = 0.1
    skin: float = 0.0
    wi: float | None = None

    def __post_init__(self):
        if self.control not in ("rate", "bhp"):
            raise ValueError(f"well {self.name}: control must be 'rate' or 'bhp'")

    def index(self, grid: Grid) -> float:
        if self.wi is not None:
            return self.wi
        return peaceman_wi(grid, self.cell, self.rw, self.skin)


def peaceman_r0(dx, dy, kx, ky):
    """Peaceman equivalent radius for an anisotropic block.

    >>> round(float(peaceman_r0(10.0, 10.0, 1.0, 1.0)), 4)   # isotropic: 0.198 dx
    1.9799
    """
    ryx = np.sqrt(ky / kx)
    rxy = np.sqrt(kx / ky)
    return 0.28 * np.sqrt(ryx * dx**2 + rxy * dy**2) / (ryx**0.5 + rxy**0.5)


def peaceman_wi(grid: Grid, cell: int, rw: float, skin: float = 0.0) -> float:
    kx, ky = grid.kx[cell], grid.ky[cell]
    r0 = peaceman_r0(grid.dx, grid.dy, kx, ky)
    if r0 <= rw:
        raise ValueError(f"wellbore radius {rw} m is not smaller than Peaceman r0 = {r0:.3g} m")
    return 2.0 * np.pi * np.sqrt(kx * ky) * grid.thickness / (np.log(r0 / rw) + skin)


def face_wi(grid: Grid, face: str) -> float:
    """Half-cell transmissibility from a 1D end cell to its outer face.

    Using this as the WI of an end-cell producer makes a BHP well behave
    like a constant-pressure outlet face.
    """
    _, T = grid.boundary_cells(face)
    if len(T) != 1:
        raise ValueError("face_wi is meant for 1D grids (one cell per face)")
    return float(T[0])


def line_source_drawdown(r, t, q, mu, k, h, phi, ct):
    """Theis / line-source pressure drop for a well producing ``q`` > 0 from
    an infinite reservoir:  dp = q mu / (4 pi k h) * E1(phi mu ct r^2 / (4 k t)).
    """
    from scipy.special import exp1

    return q * mu / (4.0 * np.pi * k * h) * exp1(phi * mu * ct * np.asarray(r) ** 2 / (4.0 * k * t))
