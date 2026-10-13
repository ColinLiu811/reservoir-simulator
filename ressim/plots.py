"""Matplotlib figures for the validation studies."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .units import DAY, MILLIDARCY, PSI  # noqa: E402

# categorical slots in fixed order; analytical curves are always ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK, **kw)
    return leg


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def bl_profiles(out, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (pvi, p) in enumerate(sorted(out["profiles"].items())):
        c = SERIES[i % len(SERIES)]
        ax.plot(out["x_d"], p["numeric"], color=c, linewidth=2,
                label=f"IMPES, {pvi:.1f} PVI  (L1 {p['l1']:.4f})")
        ax.plot(out["x_d"], p["exact"], color=INK, linewidth=1.2, linestyle="--")
    ax.plot([], [], color=INK, linewidth=1.2, linestyle="--", label="Buckley-Leverett")
    _style(ax, f"Water saturation, {out['nx']}-cell IMPES vs Buckley-Leverett",
           "dimensionless distance x/L", "water saturation Sw")
    ax.set_xlim(0, 1)
    _legend(ax, loc="upper right")
    return _save(fig, path)


def bl_production(out, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(out["pvi"], out["recovery"], color=SERIES[0], linewidth=2, label="IMPES")
    a1.plot(out["pvi"], out["recovery_exact"], color=INK, linestyle="--", linewidth=1.2,
            label="Buckley-Leverett / Welge")
    _style(a1, "Oil recovery", "pore volumes injected", "recovery factor (fraction of OOIP)")
    _legend(a1, loc="lower right")
    a2.plot(out["pvi"], out["water_cut"], color=SERIES[0], linewidth=2, label="IMPES")
    a2.plot(out["pvi"], out["water_cut_exact"], color=INK, linestyle="--", linewidth=1.2,
            label="Buckley-Leverett")
    a2.axvline(out["breakthrough_exact"], color=MUTED, linewidth=0.8, linestyle=":")
    a2.annotate(f"analytic breakthrough\n{out['breakthrough_exact']:.3f} PVI",
                (out["breakthrough_exact"], 0.15), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=MUTED)
    _style(a2, "Producer water cut", "pore volumes injected", "water cut (reservoir conditions)")
    _legend(a2, loc="lower right")
    return _save(fig, path)


def convergence(conv, path):
    rows = conv["rows"]
    dx = np.array([r["dx"] for r in rows])
    err = np.array([r["l1"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(dx, err, "o-", color=SERIES[0], linewidth=2, markersize=7,
              label=f"IMPES, observed order {conv['order']:.2f}")
    for order, ls in ((1.0, "--"), (0.5, ":")):
        ref = err[-1] * (dx / dx[-1]) ** order
        ax.loglog(dx, ref, color=MUTED, linestyle=ls, linewidth=1, label=f"O(dx^{order:g})")
    for r in rows:
        ax.annotate(f"{r['nx']}", (r["dx"], r["l1"]), xytext=(5, -10), textcoords="offset points",
                    fontsize=7, color=MUTED)
    _style(ax, f"Grid convergence at {conv['pvi']} PVI", "cell size dx [m]",
           "L1 saturation error vs Buckley-Leverett")
    _legend(ax, loc="upper left")
    return _save(fig, path)


def cfl(study, path, show=(0.9, 1.5, 2.0, 3.0)):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    best = min(r["l1"] for r in study["rows"])
    i = 0
    for r in study["rows"]:
        if r["cfl"] not in show or r["sw"] is None:
            continue
        if not r["stable"]:
            tag = "out of bounds"
        elif not r["monotone"]:
            tag = "oscillating"
        elif r["l1"] > 3 * best:
            tag = f"bounded, wrong front, L1 {r['l1']:.3f}"
        else:
            tag = f"stable, L1 {r['l1']:.4f}"
        ax.plot(study["x_d"], r["sw"], color=SERIES[i % len(SERIES)], linewidth=2,
                label=f"CFL {r['cfl']:g} ({tag})")
        i += 1
    ax.plot(study["x_d"], study["exact"], color=INK, linestyle="--", linewidth=1.2,
            label="Buckley-Leverett")
    ax.axhspan(0.8, 1.05, color=GRID, alpha=0.5, linewidth=0)
    ax.axhspan(-0.05, 0.2, color=GRID, alpha=0.5, linewidth=0)
    _style(ax, f"Explicit saturation step at fixed dt = CFL x dx/max f'  ({study['pvi']} PVI)",
           "dimensionless distance x/L", "water saturation Sw")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, 1)
    _legend(ax, loc="upper right")
    return _save(fig, path)


def single_phase(sp, path):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8))
    lin = sp["linear"]
    m = lin["x"] < 800
    a1.plot(lin["x"][m], lin["numeric"][m] / 1e6, color=SERIES[0], linewidth=2, label="implicit FD")
    a1.plot(lin["x"][m], lin["exact"][m] / 1e6, color=INK, linestyle="--", linewidth=1.2,
            label="erf solution")
    _style(a1, "1D pressure step (t = 5.6 h)", "x [m]", "pressure [MPa]")
    _legend(a1, loc="lower right")

    rad = sp["radial"]
    order = np.argsort(rad["r"])
    sel = order[(rad["r"][order] > 5) & (rad["r"][order] < 700)]
    a2.plot(rad["r"][sel], rad["numeric"][sel] / 1e6, "o", color=SERIES[0], markersize=3,
            alpha=0.6, label="2D grid blocks")
    a2.plot(rad["r"][sel], rad["exact"][sel] / 1e6, color=INK, linestyle="--", linewidth=1.2,
            label="line source")
    a2.set_xscale("log")
    _style(a2, "Radial drawdown around a producer", "distance from well [m]", "pressure [MPa]")
    _legend(a2, loc="lower right")

    t = rad["bhp_times"][1:] / 3600
    a3.plot(t, rad["bhp_series"][1:] / 1e6, color=SERIES[0], linewidth=2, label="Peaceman BHP")
    a3.plot(t, rad["bhp_series_exact"][1:] / 1e6, color=INK, linestyle="--", linewidth=1.2,
            label="line source at rw")
    a3.set_xscale("log")
    _style(a3, "Bottom-hole pressure", "time [h]", "pressure [MPa]")
    _legend(a3, loc="upper right")
    return _save(fig, path)


def five_spot(fs, path):
    g, res = fs["grid"], fs["result"]
    snaps = sorted((k[1], v) for k, v in res.snapshots.items() if isinstance(k, tuple))
    ncol = 1 + len(snaps)
    fig, axes = plt.subplots(1, ncol + 1, figsize=(3.3 * (ncol + 1), 3.4))
    ext = (0, g.lx, 0, g.ly)
    im = axes[0].imshow(np.log10(g.reshape(g.kx) / MILLIDARCY), origin="lower", extent=ext,
                        cmap="Oranges")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046)
    cb.set_label("log10 k [md]", fontsize=8, color=MUTED)
    axes[0].set_title("Permeability", fontsize=10, loc="left", color=INK)
    for ax, (pvi, (p, sw)) in zip(axes[1:ncol], snaps):
        im = ax.imshow(g.reshape(sw), origin="lower", extent=ext, cmap="Blues", vmin=0.2, vmax=0.8)
        ax.set_title(f"Sw at {pvi:.1f} PVI", fontsize=10, loc="left", color=INK)
    cb = fig.colorbar(im, ax=axes[ncol - 1], fraction=0.046)
    cb.set_label("Sw", fontsize=8, color=MUTED)
    for ax in axes[:ncol]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([g.dx / 2], [g.dy / 2], "v", color=INK, markersize=7)
        ax.plot([g.lx - g.dx / 2], [g.ly - g.dy / 2], "o", color=INK, markersize=7)
    a = axes[-1]
    a.plot(res.pvi, res.water_cut, color=SERIES[0], linewidth=2, label="water cut")
    a.plot(res.pvi, res.recovery_factor, color=SERIES[1], linewidth=2, label="recovery factor")
    _style(a, "Producer", "PVI", "fraction")
    _legend(a, loc="lower right")
    return _save(fig, path)


def solver_benchmark(rows, path):
    cells = sorted({r["cells"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs, ys = [], []
        for j, n in enumerate(cells):
            hit = [r for r in rows if r["method"] == m and r["cells"] == n]
            if hit:
                xs.append(j + (i - (len(methods) - 1) / 2) * w)
                ys.append(hit[0]["seconds"] * 1e3)
        ax.bar(xs, ys, width=w * 0.92, color=SERIES[i % len(SERIES)], label=m)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cells)), [f"{n:,} cells" for n in cells])
    _style(ax, "Pressure solve time, heterogeneous five-spot matrix", None, "wall time [ms]")
    _legend(ax, loc="upper left", ncol=len(methods))
    return _save(fig, path)


def pressure_profile_1d(x, p, path, title="Pressure"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, np.asarray(p) / PSI, color=SERIES[0], linewidth=2)
    _style(ax, title, "x [m]", "pressure [psi]")
    return _save(fig, path)


__all__ = ["bl_profiles", "bl_production", "convergence", "cfl", "single_phase", "five_spot",
           "solver_benchmark", "pressure_profile_1d", "DAY"]
