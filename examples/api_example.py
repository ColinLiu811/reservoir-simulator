"""Minimal use of the Python API: a 1D waterflood in field units,
compared with the Buckley-Leverett solution."""

import numpy as np

from ressim.buckley_leverett import BuckleyLeverett
from ressim.grid import Grid
from ressim.impes import IMPESSimulator
from ressim.relperm import CoreyRelPerm, TwoPhaseFluid
from ressim.units import CP, FT, MILLIDARCY, PSI, STB_PER_DAY
from ressim.wells import Well, face_wi

grid = Grid(nx=200, lx=1000 * FT, ly=100 * FT, thickness=30 * FT,
            perm=100 * MILLIDARCY, poro=0.2)
relperm = CoreyRelPerm(swc=0.2, sor=0.2, krw0=0.3, kro0=0.8, nw=2, no=2)
fluid = TwoPhaseFluid(mu_w=1 * CP, mu_o=5 * CP)

wells = [
    Well("INJ", cell=0, control="rate", value=100 * STB_PER_DAY),
    Well("PROD", cell=grid.n - 1, control="bhp", value=3000 * PSI, wi=face_wi(grid, "xmax")),
]
sim = IMPESSimulator(grid, fluid, relperm, sw_init=0.2, wells=wells, p_init=3000 * PSI)
res = sim.run(pvi_end=1.0, report_pvi=(0.3,))

bl = BuckleyLeverett(relperm, fluid)
sw = res.snapshots[("pvi", 0.3)][1]
err = np.mean(np.abs(sw - bl.profile(grid.x / grid.lx, 0.3)))
print(f"{res.n_steps} steps, L1 error at 0.3 PVI = {err:.4f}")
print(f"recovery at 1 PVI: {res.recovery_factor[-1]:.3f} (analytic {bl.outlet(1.0).recovery_factor:.3f})")
