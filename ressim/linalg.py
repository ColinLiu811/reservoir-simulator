"""Sparse assembly and linear solvers for the pressure equation.

The pressure matrices built here are symmetric M-matrices (a weighted graph
Laplacian plus non-negative diagonal terms from accumulation, wells and
Dirichlet boundaries), so they are symmetric positive definite whenever at
least one diagonal term pins the pressure level. That lets us use CG.

Available methods:

``direct``    sparse LU (SuperLU) - robust reference
``cg``        conjugate gradient with Jacobi (diagonal) preconditioning
``cg-ssor``   conjugate gradient with symmetric Gauss-Seidel (SSOR, w=1)
``bicgstab``  BiCGSTAB with incomplete LU
``gmres``     restarted GMRES with incomplete LU
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

METHODS = ("direct", "cg", "cg-ssor", "bicgstab", "gmres")


@dataclass
class SolveInfo:
    method: str
    iterations: int
    residual: float      # ||b - Ax|| / ||b||
    seconds: float


class LinearSolveError(RuntimeError):
    pass


def assemble_laplacian(n, a, b, coef, diag=None):
    """Build the matrix of ``sum_j coef_ij (x_i - x_j) + diag_i x_i``.

    ``a``, ``b``, ``coef`` list each connection once.

    >>> A = assemble_laplacian(3, np.array([0, 1]), np.array([1, 2]),
    ...                        np.array([1.0, 2.0]), diag=np.array([1.0, 0, 0]))
    >>> A.toarray().tolist()
    [[2.0, -1.0, 0.0], [-1.0, 3.0, -2.0], [0.0, -2.0, 2.0]]
    """
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([b, a, a, b])
    vals = np.concatenate([-coef, -coef, coef, coef])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    if diag is not None:
        A = A + sp.diags(diag)
    A.sum_duplicates()
    return A


def solve(A, b, method="direct", tol=1e-10, maxiter=None, x0=None) -> tuple[np.ndarray, SolveInfo]:
    """Solve ``A x = b`` and report iterations, relative residual and time.

    Iterative methods are run on the Jacobi-scaled system
    ``D^-1/2 A D^-1/2 y = D^-1/2 b`` because cell transmissibilities in SI
    units are tiny (~1e-15) and vary over orders of magnitude.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    A = sp.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    t0 = time.perf_counter()
    bnorm = np.linalg.norm(b) or 1.0

    if method == "direct":
        x = spla.spsolve(A.tocsc(), b)
        iters = 1
    else:
        d = A.diagonal()
        if np.any(d <= 0):
            raise LinearSolveError("matrix has a non-positive diagonal; is the pressure level pinned?")
        s = 1.0 / np.sqrt(d)
        S = sp.diags(s)
        As = (S @ A @ S).tocsr()
        bs = s * b
        y0 = None if x0 is None else x0 / s
        counter = _Counter()
        maxiter = maxiter or 10 * A.shape[0]
        if method == "cg":
            y, flag = spla.cg(As, bs, x0=y0, rtol=tol, maxiter=maxiter, callback=counter)
        elif method == "cg-ssor":
            # ILU is not symmetric, so CG gets SSOR instead:
            # M = (D + L) D^-1 (D + U), which is SPD for an SPD matrix.
            M = _ssor_preconditioner(As)
            y, flag = spla.cg(As, bs, x0=y0, rtol=tol, maxiter=maxiter, M=M, callback=counter)
        else:
            # symmetric minimum-degree ordering, no pivoting: default COLAMD +
            # partial pivoting gives unstable factors on these Laplacians
            ilu = spla.spilu(As.tocsc(), drop_tol=1e-4, fill_factor=10,
                             permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
            M = spla.LinearOperator(As.shape, ilu.solve)
            if method == "bicgstab":
                y, flag = spla.bicgstab(As, bs, x0=y0, rtol=tol, maxiter=maxiter, M=M, callback=counter)
            else:
                y, flag = spla.gmres(As, bs, x0=y0, rtol=tol, maxiter=maxiter, M=M,
                                     restart=50, callback=counter, callback_type="pr_norm")
        if flag != 0:
            raise LinearSolveError(f"{method} did not converge (flag={flag})")
        x = s * y
        iters = counter.n

    res = np.linalg.norm(b - A @ x) / bnorm
    return x, SolveInfo(method, iters, float(res), time.perf_counter() - t0)


def _ssor_preconditioner(A):
    lower = sp.tril(A, format="csr")      # D + L
    upper = sp.triu(A, format="csr")      # D + U
    d = A.diagonal()

    def apply(r):
        y = spla.spsolve_triangular(lower, r, lower=True)
        return spla.spsolve_triangular(upper, d * y, lower=False)

    return spla.LinearOperator(A.shape, apply)


class _Counter:
    def __init__(self):
        self.n = 0

    def __call__(self, *_):
        self.n += 1
