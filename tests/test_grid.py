import numpy as np
import pytest

from ressim.grid import Grid, lognormal_perm


def test_cell_numbering_and_coordinates():
    g = Grid(nx=4, ny=3, lx=8.0, ly=3.0)
    assert g.n == 12
    assert g.index(2, 1) == 6
    assert np.allclose(g.x, [1, 3, 5, 7])
    assert g.cell_at(5.1, 2.5) == g.index(2, 2)


def test_connection_count_2d():
    g = Grid(nx=5, ny=4)
    c = g.connections()
    assert len(c) == (5 - 1) * 4 + 5 * (4 - 1)
    assert np.all(c.a < c.b)


def test_series_layers_give_harmonic_effective_perm():
    # two cells in series: flux through them = k_eff * A * dp / L
    k = np.array([1e-13, 4e-13])
    g = Grid(nx=2, lx=2.0, thickness=1.0, perm=k)
    T = g.connections().T[0]
    k_eff = T * g.dx / (g.dy * g.thickness)
    assert k_eff == pytest.approx(2 * k[0] * k[1] / k.sum())


def test_boundary_half_cell_transmissibility():
    g = Grid(nx=10, ny=2, lx=10.0, ly=2.0, thickness=3.0, perm=2e-13)
    cells, T = g.boundary_cells("xmax")
    assert cells.tolist() == [9, 19]
    assert np.allclose(T, 2 * 2e-13 * (1.0 * 3.0) / 1.0)


def test_vertical_grid_depth():
    g = Grid(nx=2, ny=3, ly=30.0, vertical=True, top=1000.0)
    assert np.allclose(g.reshape(g.depth)[:, 0], [1005, 1015, 1025])


def test_bad_inputs():
    with pytest.raises(ValueError):
        Grid(nx=3, perm=[1.0, 2.0])
    with pytest.raises(ValueError):
        Grid(nx=3, poro=1.5)


def test_lognormal_field_statistics():
    k = lognormal_perm(200, 200, mean=1e-13, std_log=0.5, corr_length=5.0, seed=1)
    assert k.shape == (200, 200)
    assert np.exp(np.log(k).mean()) == pytest.approx(1e-13, rel=1e-6)
    assert np.log(k).std() == pytest.approx(0.5, rel=1e-6)
