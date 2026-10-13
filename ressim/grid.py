"""Block-centred Cartesian grids and two-point flux transmissibilities.

Cells are numbered with ``x`` fastest: ``cell = i + nx * j``. A 1D grid is
a 2D grid with ``ny = 1``. The second axis can be either areal (``y``) or
vertical (``z``, depth increasing with ``j``), which only matters when
gravity is switched on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACES = ("xmin", "xmax", "ymin", "ymax")


@dataclass
class Connections:
    """Cell-to-cell connections of a grid (each pair listed once).

    ``T`` is the geometric transmissibility k*A/L in m^3 (harmonic average of
    the two half-cell values), so the volumetric flux of a phase with
    mobility ``lam`` from ``a`` to ``b`` is ``T * lam * (Phi_a - Phi_b)``.
    """

    a: np.ndarray
    b: np.ndarray
    T: np.ndarray

    def __len__(self):
        return len(self.T)


@dataclass
class Grid:
    """Uniform Cartesian grid with heterogeneous permeability and porosity.

    Parameters
    ----------
    nx, ny : number of cells along each axis (``ny=1`` for 1D)
    lx, ly : domain size along each axis [m]
    thickness : size of the third (inactive) dimension [m]
    perm : scalar or array of permeability [m^2]; arrays may be shaped
        ``(ny, nx)`` or flat. ``perm_y`` defaults to ``perm``.
    poro : scalar or array of porosity [-]
    vertical : if True the second axis is depth (``z``) for gravity
    top : depth of the top of the model [m]
    """

    nx: int
    ny: int = 1
    lx: float = 1.0
    ly: float = 1.0
    thickness: float = 1.0
    perm: np.ndarray | float = 1e-13
    perm_y: np.ndarray | float | None = None
    poro: np.ndarray | float = 0.2
    vertical: bool = False
    top: float = 0.0
    _conn: Connections | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.nx < 1 or self.ny < 1:
            raise ValueError("grid needs at least one cell along each axis")
        self.kx = self._field(self.perm, "perm")
        self.ky = self._field(self.perm if self.perm_y is None else self.perm_y, "perm_y")
        self.phi = self._field(self.poro, "poro")
        if np.any(self.kx <= 0) or np.any(self.ky <= 0):
            raise ValueError("permeability must be positive")
        if np.any(self.phi <= 0) or np.any(self.phi >= 1):
            raise ValueError("porosity must be in (0, 1)")

    def _field(self, value, name):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n, float(arr))
        if arr.size != self.n:
            raise ValueError(f"{name} has {arr.size} values, grid has {self.n} cells")
        return arr.reshape(-1).copy()

    # ------------------------------------------------------------------ geometry
    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.thickness

    @property
    def pore_volume(self) -> np.ndarray:
        return self.phi * self.cell_volume

    @property
    def x(self) -> np.ndarray:
        """Cell-centre x coordinates (length ``nx``)."""
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Cell-centre coordinates along the second axis (length ``ny``)."""
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def depth(self) -> np.ndarray:
        """Depth of every cell centre (flat). Zero-gradient for areal grids."""
        if not self.vertical:
            return np.full(self.n, self.top)
        return np.repeat(self.top + self.y, self.nx)

    def index(self, i: int, j: int = 0) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise IndexError(f"cell ({i}, {j}) outside {self.nx}x{self.ny} grid")
        return i + self.nx * j

    def cell_at(self, x: float, y: float = 0.0) -> int:
        """Index of the cell containing point (x, y)."""
        i = min(int(x / self.dx), self.nx - 1)
        j = min(int(y / self.dy), self.ny - 1)
        return self.index(i, j)

    def reshape(self, values: np.ndarray) -> np.ndarray:
        """Turn a flat cell array into ``(ny, nx)`` for plotting."""
        return np.asarray(values).reshape(self.ny, self.nx)

    # --------------------------------------------------------- transmissibility
    def connections(self) -> Connections:
        """All interior connections with harmonic-averaged transmissibility.

        >>> g = Grid(nx=3, lx=3.0, perm=[1.0, 2.0, 2.0])
        >>> c = g.connections()
        >>> c.a.tolist(), c.b.tolist()
        ([0, 1], [1, 2])
        >>> np.round(c.T, 4).tolist()   # 2*k1*k2/(k1+k2) * A / dx
        [1.3333, 2.0]
        """
        if self._conn is None:
            idx = np.arange(self.n).reshape(self.ny, self.nx)
            ax = self.dy * self.thickness
            ay = self.dx * self.thickness
            kx = self.kx.reshape(self.ny, self.nx)
            ky = self.ky.reshape(self.ny, self.nx)

            a_x, b_x = idx[:, :-1].ravel(), idx[:, 1:].ravel()
            t_x = _harmonic(kx[:, :-1], kx[:, 1:]).ravel() * ax / self.dx
            a_y, b_y = idx[:-1, :].ravel(), idx[1:, :].ravel()
            t_y = _harmonic(ky[:-1, :], ky[1:, :]).ravel() * ay / self.dy

            self._conn = Connections(
                a=np.concatenate([a_x, a_y]),
                b=np.concatenate([b_x, b_y]),
                T=np.concatenate([t_x, t_y]),
            )
        return self._conn

    def boundary_cells(self, face: str) -> tuple[np.ndarray, np.ndarray]:
        """Cells on a domain face and their half-cell transmissibility to it.

        The half-cell value ``k * A / (d/2)`` connects the cell centre to a
        boundary held at a prescribed pressure.
        """
        idx = np.arange(self.n).reshape(self.ny, self.nx)
        kx = self.kx.reshape(self.ny, self.nx)
        ky = self.ky.reshape(self.ny, self.nx)
        if face in ("xmin", "xmax"):
            col = 0 if face == "xmin" else -1
            area = self.dy * self.thickness
            return idx[:, col].copy(), 2.0 * kx[:, col] * area / self.dx
        if face in ("ymin", "ymax"):
            row = 0 if face == "ymin" else -1
            area = self.dx * self.thickness
            return idx[row, :].copy(), 2.0 * ky[row, :] * area / self.dy
        raise ValueError(f"face must be one of {FACES}, got {face!r}")


def _harmonic(a, b):
    return 2.0 * a * b / (a + b)


def lognormal_perm(nx, ny, mean, std_log, corr_length=0.0, seed=0, dx=1.0, dy=1.0):
    """Spatially correlated log-normal permeability field, shape ``(ny, nx)``.

    White noise is smoothed with a Gaussian kernel of ``corr_length`` [m],
    re-standardised, then exponentiated so the field's geometric mean is
    ``mean`` and ``ln k`` has standard deviation ``std_log``.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((ny, nx))
    if corr_length > 0:
        z = gaussian_filter(z, sigma=(corr_length / dy, corr_length / dx), mode="reflect")
        z = (z - z.mean()) / z.std()
    return mean * np.exp(std_log * z)
