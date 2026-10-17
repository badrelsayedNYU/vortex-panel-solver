"""Symmetric airfoil at zero incidence, and basic solver sanity (spec §5.5, milestone 2)."""

import numpy as np
import pytest

from panel.geometry import naca4
from panel.solver import PanelSolver


@pytest.mark.parametrize("code", ["0006", "0012", "0018"])
@pytest.mark.parametrize("closed_te", [False, True])
def test_symmetric_alpha0_gives_zero_lift(code, closed_te):
    res = PanelSolver(naca4(code, n_panels=160, closed_te=closed_te)).solve(0.0)
    assert abs(res.cl_kj) < 1e-13 and abs(res.cl_p) < 1e-13 and abs(res.cm_c4) < 1e-13
    np.testing.assert_allclose(res.Cp, res.Cp[::-1], atol=1e-12)  # lower/upper mirror images
    np.testing.assert_allclose(res.gamma, -res.gamma[::-1], atol=1e-13)


@pytest.mark.parametrize("alpha", [-3.0, 0.0, 5.0])
def test_tangency_and_kutta_satisfied(alpha):
    s = PanelSolver(naca4("2412", n_panels=80, closed_te=True))
    res = s.solve(alpha)
    np.testing.assert_allclose(s.An[:-1] @ res.gamma, np.sin(s.theta - np.deg2rad(alpha)), atol=1e-12)
    assert abs(res.gamma[0] + res.gamma[-1]) < 1e-13


def test_cp_is_physical_naca0012():
    """Closed-TE NACA 0012 at alpha = 0: Cp_min close to the classical -0.41 near x/c = 0.1-0.15
    (Abbott & von Doenhoff give (v/V)^2_max = 1.41); stagnation at the LE (Cp -> 1);
    pressure recovery to positive Cp at the TE."""
    res = PanelSolver(naca4("0012", n_panels=320, closed_te=True)).solve(0.0)
    i = np.argmin(res.Cp)
    assert res.Cp[i] == pytest.approx(-0.41, abs=0.01)
    assert 0.05 < res.xc[i] < 0.20
    assert res.Cp.max() > 0.99
    assert res.Cp[0] > 0 and res.Cp[-1] > 0


def test_open_trailing_edge_artifact_is_confined_to_te_panels():
    """With the open (standard) NACA TE the unpaneled gap gives a spurious suction peak on the
    two TE panels only; elsewhere Cp matches the closed-TE section closely."""
    open_ = PanelSolver(naca4("0012", n_panels=160)).solve(0.0)
    closed = PanelSolver(naca4("0012", n_panels=160, closed_te=True)).solve(0.0)
    assert open_.Cp[0] < -1.0 and open_.Cp[-1] < -1.0
    interior = slice(10, -10)
    np.testing.assert_allclose(open_.Cp[interior], closed.Cp[interior], atol=0.02)


def test_lu_reuse_matches_fresh_solve():
    af = naca4("2412", n_panels=100, closed_te=True)
    s = PanelSolver(af)
    for alpha in (-4.0, 2.0, 9.0):
        g_lu = s.solve(alpha).gamma
        g_fresh = np.linalg.solve(s.An, s.rhs(np.deg2rad(alpha)))
        # two different factorizations agree to round-off: ~ cond(An) * eps * |gamma|
        tol = 10 * np.linalg.cond(s.An) * np.finfo(float).eps * np.abs(g_fresh).max()
        np.testing.assert_allclose(g_lu, g_fresh, rtol=0, atol=tol)
