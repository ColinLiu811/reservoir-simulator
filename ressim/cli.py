"""Command line interface: ``ressim {buckley-leverett,waterflood,five-spot,validate}``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import plots
from . import validation as v
from .impes import IMPESOptions
from .linalg import METHODS
from .relperm import CapillaryPressure, CoreyRelPerm, TwoPhaseFluid
from .units import CP, MILLIDARCY, PSI


def _add_rock_fluid(p):
    g = p.add_argument_group("rel-perm and fluids")
    g.add_argument("--swc", type=float, default=0.2)
    g.add_argument("--sor", type=float, default=0.2)
    g.add_argument("--krw0", type=float, default=0.3)
    g.add_argument("--kro0", type=float, default=0.8)
    g.add_argument("--nw", type=float, default=2.0)
    g.add_argument("--no", type=float, default=2.0)
    g.add_argument("--mu-w", type=float, default=1.0, help="water viscosity [cp]")
    g.add_argument("--mu-o", type=float, default=5.0, help="oil viscosity [cp]")


def _case(args):
    rp = CoreyRelPerm(args.swc, args.sor, args.krw0, args.kro0, args.nw, args.no)
    fl = TwoPhaseFluid(args.mu_w * CP, args.mu_o * CP)
    kw = {"relperm": rp, "fluid": fl}
    if hasattr(args, "perm"):
        kw["perm"] = args.perm * MILLIDARCY
        kw["days_per_pv"] = args.days_per_pv
    return v.WaterfloodCase(**kw)


def cmd_bl(args):
    case = _case(args)
    bl = case.analytic()
    print(f"Front saturation Swf     : {bl.sw_front:.4f}")
    print(f"Front water cut f(Swf)   : {float(bl.f(bl.sw_front)):.4f}")
    print(f"Shock speed df/dS        : {bl.shock_speed:.4f}")
    print(f"Breakthrough             : {bl.breakthrough_pvi:.4f} PVI")
    print()
    print(f"{'PVI':>6} {'Sw outlet':>10} {'water cut':>10} {'Sw avg':>8} {'RF':>7}")
    for t in args.pvi:
        s = bl.outlet(t)
        print(f"{t:6.2f} {s.sw_outlet:10.4f} {s.water_cut:10.4f} {s.sw_avg:8.4f} {s.recovery_factor:7.4f}")


def cmd_waterflood(args):
    case = _case(args)
    pc = None
    if args.pc_entry > 0:
        pc = CapillaryPressure(pe=args.pc_entry * PSI, swc=args.swc, sor=args.sor)
    opts = IMPESOptions(cfl=args.cfl, solver=args.solver)
    g, sim = case.simulator(args.nx, opts, capillary=pc)
    t0 = time.perf_counter()
    res = sim.run(pvi_end=args.pvi, report_pvi=args.profiles)
    secs = time.perf_counter() - t0
    bl = case.analytic()
    bt = float(res.pvi[np.argmax(res.water_cut > 0.01)]) if np.any(res.water_cut > 0.01) else np.nan
    print(f"{args.nx} cells, {res.n_steps} IMPES steps in {secs:.2f} s "
          f"(mean dt {np.mean(res.dt) / 86400:.3f} d)")
    print(f"Breakthrough      : {bt:.4f} PVI  (Buckley-Leverett {bl.breakthrough_pvi:.4f})")
    rf_exact = bl.outlet(res.pvi[-1]).recovery_factor
    print(f"Recovery at {res.pvi[-1]:.2f} PVI: {res.recovery_factor[-1]:.4f}  (analytic {rf_exact:.4f})")
    print(f"Mass balance error: {res.max_mass_balance_error:.2e}")
    print(f"Saturation bounds : max violation {res.max_bound_violation:.2e}")
    for t in args.profiles:
        sw = res.snapshots[("pvi", t)][1]
        print(f"L1 vs BL at {t:.2f} PVI: {v.l1_error(sw, bl.profile(g.x / g.lx, t)):.5f}")
    if args.plot:
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        xd = g.x / g.lx
        data = {
            "nx": args.nx, "x_d": xd,
            "profiles": {t: {"numeric": res.snapshots[("pvi", t)][1], "exact": bl.profile(xd, t),
                             "l1": v.l1_error(res.snapshots[("pvi", t)][1], bl.profile(xd, t))}
                         for t in args.profiles},
            "pvi": res.pvi, "water_cut": res.water_cut, "recovery": res.recovery_factor,
            "water_cut_exact": bl.performance(res.pvi)[0],
            "recovery_exact": bl.performance(res.pvi)[1],
            "breakthrough_exact": bl.breakthrough_pvi,
        }
        print("wrote", plots.bl_profiles(data, out / "profiles.png"))
        print("wrote", plots.bl_production(data, out / "production.png"))


def cmd_five_spot(args):
    case = _case(args)
    out = v.five_spot(n=args.n, size=args.size, heterogeneous=not args.homogeneous,
                      pvi_end=args.pvi, seed=args.seed, solver=args.solver, case=case)
    res = out["result"]
    print(f"{args.n}x{args.n} grid, {res.n_steps} IMPES steps in {out['seconds']:.1f} s, "
          f"{res.linear_iterations} linear iterations")
    print(f"Breakthrough : {out['breakthrough']:.4f} PVI")
    print(f"Recovery     : {res.recovery_factor[-1]:.4f} at {res.pvi[-1]:.2f} PVI")
    print(f"Mass balance : {res.max_mass_balance_error:.2e}")
    if args.plot:
        print("wrote", plots.five_spot(out, args.plot))


def cmd_validate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quick = args.quick

    print("== Single-phase (implicit) vs analytical")
    sp = v.single_phase_checks()
    print(f"  1D pressure step vs erf        max error {sp['linear']['max_rel_error']:.2e} of step")
    print(f"  2D drawdown vs line source     max error {sp['radial']['max_rel_error']:.2e} of drawdown")
    r = sp["radial"]
    print(f"  Peaceman BHP vs line source    {r['bhp_numeric'] / PSI:.1f} vs {r['bhp_exact'] / PSI:.1f} psi"
          f" ({abs(r['bhp_numeric'] - r['bhp_exact']) / (2e7 - r['bhp_exact']):.2%} of drawdown)")
    plots.single_phase(sp, out / "single_phase.png")

    print("\n== IMPES waterflood vs Buckley-Leverett")
    b = v.buckley_leverett_comparison(nx=100 if quick else 200)
    for t, p in sorted(b["profiles"].items()):
        print(f"  L1 saturation error at {t:.1f} PVI : {p['l1']:.4f}")
    print(f"  breakthrough   {b['breakthrough_numeric']:.4f} PVI vs {b['breakthrough_exact']:.4f}")
    print(f"  recovery       max |RF - RF_BL| = {b['recovery_error_max']:.4f}")
    print(f"  mass balance   {b['mass_balance']:.1e}")
    plots.bl_profiles(b, out / "bl_profiles.png")
    plots.bl_production(b, out / "bl_production.png")

    print("\n== Grid convergence (L1 at 0.3 PVI)")
    conv = v.grid_convergence(nxs=(25, 50, 100, 200) if quick else (25, 50, 100, 200, 400, 800))
    print(f"  {'nx':>5} {'dx [m]':>8} {'L1':>9} {'steps':>6}")
    for row in conv["rows"]:
        print(f"  {row['nx']:5d} {row['dx']:8.2f} {row['l1']:9.5f} {row['steps']:6d}")
    print(f"  observed order {conv['order']:.2f}")
    plots.convergence(conv, out / "convergence.png")

    print("\n== Explicit-step stability (fixed dt = CFL * dx / max f')")
    st = v.cfl_study()
    print(f"  max f' = {st['fprime_max']:.3f}, shock speed = {st['shock_speed']:.3f} "
          f"-> expected limit ~ CFL {st['effective_limit']:.2f}")
    print(f"  {'CFL':>5} {'bounded':>8} {'monotone':>9} {'L1':>8} {'overshoot':>10}")
    for row in st["rows"]:
        print(f"  {row['cfl']:5.2f} {str(row['stable']):>8} {str(row['monotone']):>9} "
              f"{row['l1']:8.4f} {row['violation']:10.2e}")
    plots.cfl(st, out / "cfl_stability.png")

    print("\n== Linear solvers on a heterogeneous five-spot pressure matrix")
    rows = v.solver_benchmark(sizes=(50, 100) if quick else (50, 100, 200))
    print(f"  {'cells':>7} {'method':>9} {'iters':>6} {'ms':>8} {'residual':>9}")
    for row in rows:
        print(f"  {row['cells']:7d} {row['method']:>9} {row['iterations']:6d} "
              f"{row['seconds'] * 1e3:8.1f} {row['residual']:9.1e}")
    plots.solver_benchmark(rows, out / "solver_benchmark.png")

    if not quick:
        print("\n== Heterogeneous quarter five-spot")
        fs = v.five_spot()
        res = fs["result"]
        print(f"  {fs['grid'].nx}x{fs['grid'].ny}: {res.n_steps} steps in {fs['seconds']:.1f} s,"
              f" breakthrough {fs['breakthrough']:.3f} PVI, RF {res.recovery_factor[-1]:.3f} at 1 PVI")
        plots.five_spot(fs, out / "five_spot.png")
    print(f"\nfigures written to {out}/")


def build_parser():
    p = argparse.ArgumentParser(prog="ressim", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("buckley-leverett", help="analytical Buckley-Leverett / Welge results")
    _add_rock_fluid(s)
    s.add_argument("--pvi", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0, 5.0])
    s.set_defaults(func=cmd_bl)

    s = sub.add_parser("waterflood", help="1D IMPES waterflood compared with Buckley-Leverett")
    _add_rock_fluid(s)
    s.add_argument("--nx", type=int, default=200)
    s.add_argument("--pvi", type=float, default=2.0, help="stop after this many PV injected")
    s.add_argument("--profiles", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4])
    s.add_argument("--perm", type=float, default=100.0, help="permeability [md]")
    s.add_argument("--days-per-pv", type=float, default=500.0, help="injection rate as days per PV")
    s.add_argument("--cfl", type=float, default=0.9)
    s.add_argument("--pc-entry", type=float, default=0.0, help="Brooks-Corey entry pressure [psi]")
    s.add_argument("--solver", choices=METHODS, default="direct")
    s.add_argument("--plot", metavar="DIR")
    s.set_defaults(func=cmd_waterflood)

    s = sub.add_parser("five-spot", help="2D quarter five-spot waterflood")
    _add_rock_fluid(s)
    s.add_argument("--n", type=int, default=40, help="cells per side")
    s.add_argument("--size", type=float, default=300.0, help="side length [m]")
    s.add_argument("--pvi", type=float, default=1.0)
    s.add_argument("--perm", type=float, default=100.0, help="(geometric) mean permeability [md]")
    s.add_argument("--days-per-pv", type=float, default=500.0)
    s.add_argument("--homogeneous", action="store_true")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--solver", choices=METHODS, default="direct")
    s.add_argument("--plot", metavar="PNG")
    s.set_defaults(func=cmd_five_spot)

    s = sub.add_parser("validate", help="run every analytical comparison and write figures")
    s.add_argument("--out", default="validation_output")
    s.add_argument("--quick", action="store_true", help="smaller grids, skip the five-spot")
    s.set_defaults(func=cmd_validate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
