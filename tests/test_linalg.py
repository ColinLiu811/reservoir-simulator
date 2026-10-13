import numpy as np
import pytest

from ressim.grid import Grid, lognormal_perm
from ressim.linalg import METHODS, LinearSolveError, assemble_laplacian, solve


def _pressure_matrix(nx=30, ny=30):
    k = lognormal_perm(nx, ny, 1e-13, 1.5, corr_length=3.0, seed=4)
    g = Grid(nx=nx, ny=ny, lx=300.0, ly=300.0, thickness=10.0, perm=k)
    c = g.connections()
    diag = np.zeros(g.n)
    diag[0] = 1e3 * c.T.mean()    # pin pressure level like a BHP well (lam ~ 1/mu)
    A = assemble_laplacian(g.n, c.a, c.b, c.T * 1e3, diag)
    return A


def test_matrix_is_symmetric_with_zero_row_sums_away_from_wells():
    A = _pressure_matrix()
    assert abs(A - A.T).max() == 0
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    assert np.allclose(rowsum[1:], 0, atol=1e-12 * abs(A.diagonal()).max())


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_agree_with_direct(method):
    A = _pressure_matrix()
    rng = np.random.default_rng(0)
    x_true = rng.uniform(1e7, 2e7, A.shape[0])
    b = A @ x_true
    x, info = solve(A, b, method=method, tol=1e-12)
    assert info.residual < 1e-9
    assert np.allclose(x, x_true, rtol=1e-6)
    if method != "direct":
        assert info.iterations > 0


def test_ssor_preconditioning_cuts_iterations():
    A = _pressure_matrix(50, 50)
    b = np.zeros(A.shape[0])
    b[-1] = 1.0
    _, jac = solve(A, b, method="cg", tol=1e-8)
    _, ssor = solve(A, b, method="cg-ssor", tol=1e-8)
    assert ssor.iterations < jac.iterations / 2


def test_singular_pressure_matrix_is_reported():
    g = Grid(nx=4)
    c = g.connections()
    A = assemble_laplacian(g.n, c.a, c.b, c.T)
    A = A.tolil()
    A[0, 0] = 0.0
    with pytest.raises(LinearSolveError):
        solve(A.tocsr(), np.ones(4), method="cg")


def test_unknown_method():
    with pytest.raises(ValueError):
        solve(np.eye(2), np.ones(2), method="jacobi")
