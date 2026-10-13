"""Conversion factors from oilfield units to the SI units used internally.

Multiply a value in the named unit by the constant to get SI:

>>> round(100 * MILLIDARCY / 1e-15, 3)   # 100 md in m^2, scaled
98.692
>>> round(1000 * PSI / 1e6, 4)           # 1000 psi in MPa
6.8948
"""

FT = 0.3048                       # m
DARCY = 9.869233e-13              # m^2
MILLIDARCY = 1e-3 * DARCY         # m^2
PSI = 6894.757293168              # Pa
BAR = 1e5                         # Pa
CP = 1e-3                         # Pa.s
DAY = 86400.0                     # s
BBL = 0.158987294928              # m^3
STB_PER_DAY = BBL / DAY           # m^3/s
PER_PSI = 1.0 / PSI               # 1/Pa
G = 9.80665                       # m/s^2


def to_si(value, unit):
    """Convert ``value`` given in ``unit`` to SI.

    >>> round(to_si(1.0, "ft"), 4)
    0.3048
    """
    factors = {
        "m": 1.0, "ft": FT,
        "m2": 1.0, "darcy": DARCY, "md": MILLIDARCY,
        "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": BAR, "psi": PSI,
        "pa.s": 1.0, "cp": CP,
        "s": 1.0, "day": DAY,
        "m3/s": 1.0, "m3/day": 1.0 / DAY, "stb/day": STB_PER_DAY, "bbl/day": STB_PER_DAY,
        "1/pa": 1.0, "1/psi": PER_PSI,
    }
    try:
        return value * factors[unit.lower()]
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; known: {sorted(factors)}") from None


def from_si(value, unit):
    """Inverse of :func:`to_si`.

    >>> round(from_si(PSI * 2500, "psi"), 6)
    2500.0
    """
    return value / to_si(1.0, unit)
