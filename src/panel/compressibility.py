"""Subsonic compressibility corrections and critical Mach number (spec §4; Anderson ch. 11)."""

import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4


def prandtl_glauert(cp0, mach):
    """Cp = Cp0 / sqrt(1 - M^2)  (spec §4; Anderson §11.4)."""
    return cp0 / np.sqrt(1.0 - mach**2)


def karman_tsien(cp0, mach):
    """Cp = Cp0 / [sqrt(1 - M^2) + M^2/(1 + sqrt(1 - M^2)) * Cp0/2]  (spec §4; Anderson §11.5)."""
    beta = np.sqrt(1.0 - mach**2)
    return cp0 / (beta + mach**2 / (1.0 + beta) * cp0 / 2.0)


RULES = {"PG": prandtl_glauert, "KT": karman_tsien}


def cp_critical(mach, gamma=GAMMA):
    """Cp at which the local Mach number reaches 1  (spec §4; Anderson §11.6):
    Cp_cr = 2/(gamma M^2) * {[(2 + (gamma - 1) M^2)/(gamma + 1)]^(gamma/(gamma - 1)) - 1}."""
    return 2.0 / (gamma * mach**2) * (((2.0 + (gamma - 1.0) * mach**2) / (gamma + 1.0))
                                      ** (gamma / (gamma - 1.0)) - 1.0)


def kt_singular_mach(cp0):
    """Mach at which the Kármán–Tsien denominator vanishes for this Cp0 (< 0); 1 if none.
    Solves sqrt(1 - M^2) + M^2/(1 + sqrt(1 - M^2)) Cp0/2 = 0 for M in (0, 1)."""
    if cp0 >= 0:
        return 1.0
    f = lambda m: np.sqrt(1 - m**2) + m**2 / (1 + np.sqrt(1 - m**2)) * cp0 / 2
    return brentq(f, 1e-6, 1 - 1e-12) if f(1 - 1e-12) < 0 else 1.0


def critical_mach(cp0_min, rule="KT"):
    """Free-stream Mach where the corrected minimum pressure coefficient reaches sonic conditions:
    solve Cp_min,corrected(M) = Cp_cr(M) with brentq  (spec §4).

    Both corrections are monotonic in Cp0, so the corrected minimum is the correction of the
    incompressible minimum Cp0_min (< 0). The bracket stops just below the KT singularity, where
    the corrected Cp tends to -infinity.
    """
    if cp0_min >= 0:
        raise ValueError("Cp0_min must be negative (there is always suction somewhere)")
    correct = RULES[rule]
    upper = (kt_singular_mach(cp0_min) if rule == "KT" else 1.0) * (1 - 1e-9)
    upper = min(upper, 1 - 1e-9)
    f = lambda m: correct(cp0_min, m) - cp_critical(m)
    return brentq(f, 1e-3, upper, xtol=1e-12)
