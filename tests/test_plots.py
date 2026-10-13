from ressim import plots
from ressim import validation as v


def test_figures_render(tmp_path):
    b = v.buckley_leverett_comparison(nx=40, pvis=(0.2,), pvi_end=0.6)
    paths = [
        plots.bl_profiles(b, tmp_path / "a.png"),
        plots.bl_production(b, tmp_path / "b.png"),
        plots.convergence(v.grid_convergence(nxs=(20, 40)), tmp_path / "c.png"),
        plots.cfl(v.cfl_study(nx=30, cfls=(0.9, 3.0)), tmp_path / "d.png", show=(0.9, 3.0)),
        plots.five_spot(v.five_spot(n=8, size=80.0, pvi_end=0.3, report_pvi=(0.1,)), tmp_path / "e.png"),
        plots.solver_benchmark(v.solver_benchmark(sizes=(10,)), tmp_path / "f.png"),
    ]
    for p in paths:
        assert p.stat().st_size > 5000
