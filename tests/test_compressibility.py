"""Compressibility corrections and critical Mach (spec §4)."""

import numpy as np
import pytest

from panel.compressibility import (cp_critical, critical_mach, karman_tsien, kt_singular_mach,
                                   prandtl_glauert)


def test_corrections_reduce_to_incompressible_at_m0():
    cp0 = np.array([-2.0, -0.5, 0.3])
    np.testing.assert_allclose(prandtl_glauert(cp0, 0.0), cp0)
    np.testing.assert_allclose(karman_tsien(cp0, 0.0), cp0)


def test_karman_tsien_tends_to_prandtl_glauert_for_small_cp():
    """KT = PG to first order in Cp0; their relative gap shrinks linearly with Cp0."""
    M = 0.6
    gaps = [abs(karman_tsien(c, M) / prandtl_glauert(c, M) - 1) for c in (-0.1, -0.01, -0.001)]
    assert gaps[0] / gaps[1] == pytest.approx(10, rel=0.05)
    assert gaps[2] < 1e-3


def test_cp_critical_values():
    """Cp_cr(M = 1) = 0; Cp_cr(0.5) = -2.13 (Anderson Fig. 11.6); monotonic increasing in M."""
    assert cp_critical(1.0) == pytest.approx(0.0, abs=1e-14)
    assert cp_critical(0.5) == pytest.approx(-2.13, abs=0.005)
    M = np.linspace(0.2, 1.0, 50)
    assert np.all(np.diff(cp_critical(M)) > 0)


def test_anderson_naca0012_example():
    """Anderson, Fundamentals of Aerodynamics, ch. 11 example: NACA 0012 at alpha = 0 with
    Cp0_min = -0.43 has M_cr = 0.74 by Prandtl–Glauert."""
    assert critical_mach(-0.43, "PG") == pytest.approx(0.74, abs=0.005)


@pytest.mark.parametrize("rule", ["PG", "KT"])
@pytest.mark.parametrize("cp0", [-0.2, -0.43, -1.0, -3.0, -8.0])
def test_root_satisfies_sonic_condition(rule, cp0):
    m = critical_mach(cp0, rule)
    corrected = prandtl_glauert(cp0, m) if rule == "PG" else karman_tsien(cp0, m)
    assert corrected == pytest.approx(cp_critical(m), rel=1e-9)
    if rule == "KT":
        assert m < kt_singular_mach(cp0)


def test_stronger_suction_lowers_critical_mach_and_kt_below_pg():
    cps = [-0.2, -0.5, -1.0, -2.0, -4.0]
    for rule in ("PG", "KT"):
        m = [critical_mach(c, rule) for c in cps]
        assert np.all(np.diff(m) < 0)
    assert all(critical_mach(c, "KT") < critical_mach(c, "PG") for c in cps)
