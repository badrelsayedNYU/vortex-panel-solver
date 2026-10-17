"""Boundary-layer building blocks against flat-plate theory, and airfoil-level sanity (spec §6)."""

import numpy as np
import pytest

from panel.boundary_layer import (head_H1_from_H, head_H_from_H1, march_surface, michel_re_theta,
                                  thwaites, thwaites_H, viscous_drag)
from panel.geometry import naca4
from panel.solver import PanelSolver


def test_thwaites_flat_plate_matches_blasius():
    """U_e = 1: Thwaites gives theta = sqrt(0.45 nu x) = 0.671 sqrt(nu x); Blasius 0.664 sqrt(nu x)."""
    nu = 1e-6
    s = np.linspace(0, 1, 2001)
    theta, lam, _ = thwaites(s, np.ones_like(s), nu)
    np.testing.assert_allclose(theta[1:], np.sqrt(0.45 * nu * s[1:]), rtol=1e-10)
    assert theta[-1] / np.sqrt(nu) == pytest.approx(0.664, rel=0.015)
    np.testing.assert_allclose(lam, 0.0, atol=1e-12)  # dU_e/ds of a constant is ~1e-15 round-off
    assert thwaites_H(0.0) == pytest.approx(2.61)  # Blasius H = 2.59


def test_thwaites_stagnation_point_flow():
    """U_e = k s: lambda = 0.075 everywhere (Thwaites' stagnation value)."""
    s = np.linspace(0, 0.1, 501)
    theta, lam, _ = thwaites(s, 3.0 * s, 1e-6)
    np.testing.assert_allclose(lam, 0.075, rtol=1e-3)


def test_head_shape_factor_inverse_is_consistent():
    H = np.linspace(1.3, 2.4, 30)
    np.testing.assert_allclose(head_H_from_H1(head_H1_from_H(H)), H, rtol=0.03)


def test_michel_transition_on_flat_plate():
    """Flat-plate Re_theta = 0.671 sqrt(Re_x) meets Michel's curve near Re_x ~ 1e6 (the classical
    flat-plate transition Reynolds number is 3e5 to 3e6)."""
    re_x = np.logspace(4, 7, 4000)
    i = np.argmax(0.671 * np.sqrt(re_x) > michel_re_theta(re_x))
    assert 3e5 < re_x[i] < 3e6


def test_turbulent_flat_plate_drag_matches_power_law():
    """Flat plate at Re = 1e7 tripped at x = 0.001 (effectively fully turbulent): Squire–Young with
    U_e = 1 is 2 theta per side; compare with the 1/7-power-law C_d = 0.074 Re^-0.2 per side."""
    re = 1e7
    s = np.linspace(0, 1, 4001)
    res = march_surface(s, s, np.ones_like(s), re, x_trip=0.001)
    assert res.x_transition == pytest.approx(0.001, abs=3e-4)
    assert res.cd == pytest.approx(0.074 * re**-0.2, rel=0.15)


def test_flat_plate_natural_transition_mixed_flow():
    """Untripped flat plate at Re = 1e7: Michel transition near Re_x ~ 1e6, and the drag lies
    between the all-laminar (Blasius, 1.328/sqrt(Re) per side) and all-turbulent values."""
    re = 1e7
    s = np.linspace(0, 1, 4001)
    res = march_surface(s, s, np.ones_like(s), re)
    assert 0.03 < res.x_transition < 0.3
    assert 1.328 / np.sqrt(re) < res.cd < 0.074 * re**-0.2


def test_airfoil_drag_trends():
    s = PanelSolver(naca4("0012", n_panels=200, closed_te=True))
    r0 = viscous_drag(s.solve(0.0), 1e6)
    assert 0.004 < r0.cd < 0.010 and not r0.separated
    assert r0.upper.x_transition == pytest.approx(r0.lower.x_transition)  # symmetric
    r_hi = viscous_drag(s.solve(0.0), 1e7)
    assert r_hi.cd < r0.cd  # higher Re: thinner boundary layers
    assert r_hi.upper.x_transition < r0.upper.x_transition
    r4 = viscous_drag(s.solve(4.0), 1e6)
    assert r4.upper.x_transition < r0.upper.x_transition < r4.lower.x_transition
    assert viscous_drag(s.solve(8.0), 1e6).separated  # flagged, not hidden
