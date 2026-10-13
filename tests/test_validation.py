import numpy as np
import pytest

from ressim import validation as v


def test_bl_comparison_summary():
    out = v.buckley_leverett_comparison(nx=100, pvis=(0.2, 0.3), pvi_end=1.0)
    assert all(p["l1"] < 0.01 for p in out["profiles"].values())
    assert out["breakthrough_numeric"] == pytest.approx(out["breakthrough_exact"], rel=0.03)
    assert out["recovery_error_max"] < 0.01
    assert out["mass_balance"] < 1e-12


def test_grid_convergence_order():
    out = v.grid_convergence(nxs=(25, 50, 100, 200))
    errs = [r["l1"] for r in out["rows"]]
    assert errs[-1] < errs[0] / 3
    assert 0.5 < out["order"] < 1.2


def test_cfl_study_separates_stable_and_unstable():
    out = v.cfl_study(nx=60, cfls=(0.9, 3.0))
    small, big = out["rows"]
    assert small["stable"] and small["monotone"]
    assert not (big["stable"] and big["monotone"])
    assert out["effective_limit"] > 1.0


def test_single_phase_checks():
    out = v.single_phase_checks()
    assert out["linear"]["max_rel_error"] < 5e-3
    assert out["radial"]["max_rel_error"] < 0.03
    assert out["radial"]["bhp_numeric"] == pytest.approx(out["radial"]["bhp_exact"], rel=2e-3)


def test_five_spot_small():
    out = v.five_spot(n=12, size=120.0, pvi_end=0.5, report_pvi=(0.2,))
    res = out["result"]
    assert res.max_mass_balance_error < 1e-10
    assert 0.1 < out["breakthrough"] < 0.5
    assert ("pvi", 0.2) in res.snapshots


def test_solver_benchmark_agrees():
    rows = v.solver_benchmark(sizes=(20,))
    assert {r["method"] for r in rows} == set(v.METHODS)
    assert all(r["residual"] < 1e-7 for r in rows)
    assert all(r["max_dp_vs_first"] < 10.0 for r in rows)     # Pa, on ~2e7 Pa
