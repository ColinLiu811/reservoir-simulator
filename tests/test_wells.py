import numpy as np
import pytest

from ressim.grid import Grid
from ressim.wells import Well, face_wi, peaceman_r0, peaceman_wi


def test_peaceman_isotropic_radius():
    assert peaceman_r0(20.0, 20.0, 1.0, 1.0) == pytest.approx(0.198 * 20.0, rel=1e-3)


def test_peaceman_anisotropic_reduces_to_isotropic():
    assert peaceman_r0(10.0, 30.0, 2.0, 2.0) == pytest.approx(
        0.14 * np.sqrt(10.0**2 + 30.0**2), rel=1e-12)


def test_wi_formula():
    g = Grid(nx=5, ny=5, lx=50.0, ly=50.0, thickness=8.0, perm=1e-13)
    r0 = peaceman_r0(10.0, 10.0, 1e-13, 1e-13)
    expect = 2 * np.pi * 1e-13 * 8.0 / (np.log(r0 / 0.1) + 2.0)
    assert peaceman_wi(g, 12, 0.1, skin=2.0) == pytest.approx(expect)


def test_rw_larger_than_block_rejected():
    g = Grid(nx=2, ny=2, lx=2.0, ly=2.0)
    with pytest.raises(ValueError):
        peaceman_wi(g, 0, rw=0.5)


def test_face_wi_and_override():
    g = Grid(nx=10, lx=100.0, ly=5.0, thickness=2.0, perm=3e-13)
    wi = face_wi(g, "xmax")
    assert wi == pytest.approx(2 * 3e-13 * 10.0 / 10.0)
    assert Well("P", 9, "bhp", 1e7, wi=wi).index(g) == wi


def test_bad_control():
    with pytest.raises(ValueError):
        Well("X", 0, "pressure", 1.0)
