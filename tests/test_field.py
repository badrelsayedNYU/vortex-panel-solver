"""Field velocity checks (milestone 5)."""

import numpy as np
import pytest

from panel.field import inside, stagnation_point, velocity
from panel.geometry import naca4
from panel.solver import PanelSolver


@pytest.fixture(scope="module")
def case():
    af = naca4("2412", n_panels=200, closed_te=True)
    return af, PanelSolver(af).solve(6.0)


def test_far_field_is_free_stream(case):
    af, res = case
    u, v = velocity(af, res.gamma, 6.0, np.array([60.0, -50.0]), np.array([40.0, -60.0]))
    a = np.deg2rad(6.0)
    np.testing.assert_allclose(u, np.cos(a), atol=0.02)
    np.testing.assert_allclose(v, np.sin(a), atol=0.02)


def test_field_matches_surface_solution_just_outside(case):
    """At a small outward offset from each control point the field velocity must be tangent to
    the wall (no flow through it) and equal the surface velocity V_t."""
    af, res = case
    S, th, xc, yc = af.panels()
    eps = 1e-6
    n = np.array([-np.sin(th), np.cos(th)])  # outward normal (clockwise node ordering)
    u, v = velocity(af, res.gamma, 6.0, xc + eps * n[0], yc + eps * n[1])
    vn = u * n[0] + v * n[1]
    vt = u * np.cos(th) + v * np.sin(th)
    interior = slice(5, -5)  # skip the TE corner where the offset point sees the opposite surface
    np.testing.assert_allclose(vn[interior], 0.0, atol=1e-4)
    np.testing.assert_allclose(vt[interior], res.Vt[interior], atol=1e-4)


def test_inside_mask(case):
    af, _ = case
    m = inside(af, np.array([0.3, 0.3, 1.2]), np.array([0.02, 0.5, 0.0]))
    assert m.tolist() == [True, False, False]


def test_stagnation_point_moves_aft_on_lower_surface_with_alpha():
    s = PanelSolver(naca4("0012", n_panels=200, closed_te=True))
    x0, y0 = stagnation_point(s.solve(0.0))
    # at alpha = 0 it lies between the two control points beside the nose, i.e. at their x
    # (the midpoint of the first panel, slightly aft of x = 0 on the curved LE)
    nose_cp_x = s.xc[np.argsort(s.xc)[:2]].mean()
    assert x0 == pytest.approx(nose_cp_x, abs=1e-12) and y0 == pytest.approx(0.0, abs=1e-12)
    x8, y8 = stagnation_point(s.solve(8.0))
    assert x8 > x0 and y8 < 0
