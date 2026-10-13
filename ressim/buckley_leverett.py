"""Buckley-Leverett analytical solution for 1D incompressible waterflooding.

Without capillarity and gravity, water saturation obeys the hyperbolic
conservation law

    dS/dt_D + d f(S)/dx_D = 0,   x_D = x/L,  t_D = injected pore volumes

whose entropy solution for constant injection into a uniform initial
saturation ``Swi`` is a shock followed by a rarefaction fan. The shock
saturation ``Sf`` comes from the Welge tangent: the line from
``(Swi, f(Swi))`` that touches the fractional flow curve, i.e. the
maximiser of the secant slope (f(S) - f(Swi)) / (S - Swi). Behind the shock
each saturation travels at x_D = t_D f'(S).

After breakthrough the Welge relation gives the average saturation from
the outlet saturation S2:  S_avg = S2 + (1 - f(S2)) / f'(S2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .relperm import CoreyRelPerm, TwoPhaseFluid, fractional_flow, fractional_flow_derivative


@dataclass
class OutletState:
    pvi: float
    sw_outlet: float
    water_cut: float
    sw_avg: float
    oil_recovered_pv: float        # cumulative oil produced / pore volume
    recovery_factor: float         # fraction of initial oil in place


class BuckleyLeverett:
    """Analytical waterflood for a given rel-perm / fluid set.

    >>> rp = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8)
    >>> bl = BuckleyLeverett(rp, TwoPhaseFluid(mu_w=1e-3, mu_o=5e-3))
    >>> round(bl.sw_front, 3), round(bl.breakthrough_pvi, 3)
    (0.554, 0.445)
    """

    def __init__(self, rp: CoreyRelPerm, fluid: TwoPhaseFluid, swi: float | None = None, n_table=20001):
        self.rp = rp
        self.fluid = fluid
        self.swi = rp.swc if swi is None else float(swi)
        self.s_max = 1.0 - rp.sor
        if not rp.swc <= self.swi < self.s_max:
            raise ValueError("initial saturation must lie in [Swc, 1 - Sor)")
        self.f_i = float(self.f(self.swi))

        self.sw_front, self.shock_speed = self._welge_tangent()

        # rarefaction branch S in [Sf, 1-Sor]: f'(S) must decrease monotonically
        s = np.linspace(self.sw_front, self.s_max, n_table)
        d = fractional_flow_derivative(s, rp, fluid)
        d[0] = self.shock_speed                     # tangency: f'(Sf) = shock speed
        if np.any(np.diff(d) > 1e-9 * d.max()):
            raise ValueError("f'(S) is not monotone behind the front; fractional flow is not S-shaped")
        self._branch_s = s[::-1]                    # ascending f'
        self._branch_d = d[::-1]

    def f(self, s):
        return fractional_flow(s, self.rp, self.fluid)

    def _welge_tangent(self):
        def neg_slope(s):
            return -(self.f(s) - self.f_i) / (s - self.swi)

        s = np.linspace(self.swi, self.s_max, 4001)[1:]
        slope = -neg_slope(s)
        # ties (a straight segment of f) belong to the shock: take the largest S
        k = int(np.nonzero(slope >= slope.max() * (1 - 1e-9))[0][-1])
        if k == len(s) - 1:                         # tangent hits the end point: pure shock
            return self.s_max, float(-neg_slope(self.s_max))
        lo, hi = s[max(k - 1, 0)], s[min(k + 1, len(s) - 1)]
        res = minimize_scalar(neg_slope, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-12})
        return float(res.x), float(-res.fun)

    @property
    def breakthrough_pvi(self) -> float:
        return 1.0 / self.shock_speed

    def _s_for_speed(self, speed):
        """Saturation on the rarefaction branch with f'(S) = speed."""
        return np.interp(speed, self._branch_d, self._branch_s)

    def profile(self, x_d, pvi):
        """Water saturation at dimensionless positions ``x_d`` after ``pvi``
        pore volumes have been injected."""
        x_d = np.asarray(x_d, dtype=float)
        if pvi <= 0:
            return np.full_like(x_d, self.swi)
        speed = x_d / pvi
        return np.where(speed < self.shock_speed, self._s_for_speed(speed), self.swi)

    def front_position(self, pvi) -> float:
        return pvi * self.shock_speed

    def outlet(self, pvi) -> OutletState:
        """Production performance at x_D = 1 after ``pvi`` injected PV."""
        oip = 1.0 - self.swi
        if pvi < self.breakthrough_pvi:
            s2 = self.swi
            fw = self.f_i
            s_avg = self.swi + pvi * (1.0 - self.f_i)
        else:
            s2 = float(self._s_for_speed(1.0 / pvi))
            fw = float(self.f(s2))
            s_avg = s2 + (1.0 - fw) * pvi            # Welge: f'(S2) = 1 / pvi
        np_pv = s_avg - self.swi
        return OutletState(pvi, s2, fw, s_avg, np_pv, np_pv / oip)

    def performance(self, pvi):
        """Vectorised :meth:`outlet`: arrays of water cut and recovery factor."""
        states = [self.outlet(t) for t in np.atleast_1d(pvi)]
        return (np.array([s.water_cut for s in states]),
                np.array([s.recovery_factor for s in states]))
