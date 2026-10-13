"""Two-phase (water/oil) rock-fluid properties.

Corey relative permeability:

    Se  = (Sw - Swc) / (1 - Swc - Sor)
    krw = krw0 * Se^nw
    kro = kro0 * (1 - Se)^no

Brooks-Corey style capillary pressure (zero by default):

    Pc = pe * Se^(-1/lambda), capped at ``pc_max`` (default 10 pe)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TwoPhaseFluid:
    """Incompressible water and oil: viscosities [Pa.s], densities [kg/m^3]."""

    mu_w: float
    mu_o: float
    rho_w: float = 1000.0
    rho_o: float = 800.0


@dataclass
class CoreyRelPerm:
    swc: float = 0.2
    sor: float = 0.2
    krw0: float = 0.3
    kro0: float = 0.8
    nw: float = 2.0
    no: float = 2.0

    def __post_init__(self):
        if not 0 <= self.swc < 1 - self.sor <= 1:
            raise ValueError("need 0 <= swc < 1 - sor")

    def se(self, sw):
        s = (np.asarray(sw, dtype=float) - self.swc) / (1.0 - self.swc - self.sor)
        return np.clip(s, 0.0, 1.0)

    def krw(self, sw):
        return self.krw0 * self.se(sw) ** self.nw

    def kro(self, sw):
        return self.kro0 * (1.0 - self.se(sw)) ** self.no

    def dse_dsw(self):
        return 1.0 / (1.0 - self.swc - self.sor)


@dataclass
class CapillaryPressure:
    """Brooks-Corey drainage curve. ``pe = 0`` switches capillarity off."""

    pe: float = 0.0          # entry pressure [Pa]
    lam: float = 2.0         # pore-size distribution index
    pc_max: float | None = None   # cap as Se -> 0 [Pa]; None = 10 * pe
    swc: float = 0.2
    sor: float = 0.2

    def se(self, sw):
        s = (np.asarray(sw, dtype=float) - self.swc) / (1.0 - self.swc - self.sor)
        return np.clip(s, 1e-6, 1.0)

    def __call__(self, sw):
        if self.pe == 0.0:
            return np.zeros_like(np.asarray(sw, dtype=float))
        cap = 10.0 * self.pe if self.pc_max is None else self.pc_max
        return np.minimum(self.pe * self.se(sw) ** (-1.0 / self.lam), cap)

    def derivative(self, sw, h=1e-6):
        """dPc/dSw by central difference (<= 0)."""
        sw = np.asarray(sw, dtype=float)
        return (self(sw + h) - self(sw - h)) / (2.0 * h)


def mobilities(sw, rp: CoreyRelPerm, fluid: TwoPhaseFluid):
    """Water and oil mobilities kr/mu [1/(Pa.s)]."""
    return rp.krw(sw) / fluid.mu_w, rp.kro(sw) / fluid.mu_o


def fractional_flow(sw, rp: CoreyRelPerm, fluid: TwoPhaseFluid):
    """Water fractional flow fw = lam_w / (lam_w + lam_o), no gravity or Pc.

    >>> rp = CoreyRelPerm(swc=0.0, sor=0.0, krw0=1.0, kro0=1.0, nw=1, no=1)
    >>> float(fractional_flow(0.5, rp, TwoPhaseFluid(1.0, 1.0)))
    0.5
    """
    lw, lo = mobilities(sw, rp, fluid)
    return lw / (lw + lo)


def fractional_flow_derivative(sw, rp: CoreyRelPerm, fluid: TwoPhaseFluid, h=1e-7):
    sw = np.asarray(sw, dtype=float)
    lo = np.clip(sw - h, rp.swc, 1 - rp.sor)
    hi = np.clip(sw + h, rp.swc, 1 - rp.sor)
    return (fractional_flow(hi, rp, fluid) - fractional_flow(lo, rp, fluid)) / (hi - lo)


def max_dfw_dsw(rp: CoreyRelPerm, fluid: TwoPhaseFluid, n=2001):
    """Largest characteristic speed df/dS over the mobile range (for CFL)."""
    s = np.linspace(rp.swc, 1 - rp.sor, n)
    return float(np.max(fractional_flow_derivative(s, rp, fluid)))


def mobility_ratio(rp: CoreyRelPerm, fluid: TwoPhaseFluid):
    """End-point mobility ratio M = (krw0/mu_w) / (kro0/mu_o)."""
    return (rp.krw0 / fluid.mu_w) / (rp.kro0 / fluid.mu_o)
