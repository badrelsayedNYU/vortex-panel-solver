"""Geometry tests (spec §2, milestone 1)."""

import numpy as np
import pytest

from panel.geometry import (KarmanTrefftz, load_selig, naca4, naca4_camber, naca4_thickness,
                            signed_area, write_selig)


@pytest.mark.parametrize("code", ["0012", "2412", "4415"])
def test_naca_ordering_and_counts(code):
    af = naca4(code, n_panels=160)
    assert af.X.size == 161 and af.n_panels == 160
    assert signed_area(af.X, af.Y) < 0, "nodes must run clockwise"
    assert af.X[0] == pytest.approx(1.0, abs=1e-3) and af.X[-1] == pytest.approx(1.0, abs=1e-3)
    assert af.Y[0] <= af.Y[-1], "first node must be the lower-surface TE"
    assert af.X[80] == pytest.approx(0.0, abs=1e-12), "middle node is the LE"


def test_naca0012_max_thickness():
    """Max thickness 0.12c near x = 0.30 (Abbott & von Doenhoff)."""
    x = np.linspace(0, 1, 200001)
    yt = naca4_thickness(x, 0.12)
    assert 2 * yt.max() == pytest.approx(0.12, abs=2e-4)
    assert x[np.argmax(yt)] == pytest.approx(0.30, abs=0.005)


def test_trailing_edge_gap_open_vs_closed():
    open_, closed = naca4("0012", closed_te=False), naca4("0012", closed_te=True)
    # open TE: total gap = 2 * y_t(1) = 10 t (0.2969 - 0.1260 - 0.3516 + 0.2843 - 0.1015)
    assert open_.Y[-1] - open_.Y[0] == pytest.approx(2 * 5 * 0.12 * 0.0021, rel=1e-10)
    assert closed.Y[-1] - closed.Y[0] == pytest.approx(0.0, abs=1e-6)


def test_naca2412_camber_line():
    """Max camber 0.02 at x = 0.4, continuous slope at x = p, zero camber at both ends."""
    x = np.array([0.0, 0.4, 1.0])
    yc, dyc = naca4_camber(x, 0.02, 0.4)
    np.testing.assert_allclose(yc, [0.0, 0.02, 0.0], atol=1e-15)
    assert dyc[1] == pytest.approx(0.0, abs=1e-15)


def test_symmetric_section_is_mirror_symmetric():
    af = naca4("0012", n_panels=100)
    np.testing.assert_allclose(af.X, af.X[::-1], atol=1e-15)
    np.testing.assert_allclose(af.Y, -af.Y[::-1], atol=1e-15)


def test_selig_round_trip(tmp_path):
    """Write a dense NACA 2412 in Selig order, reload and repanel at N = 160: every new node
    must lie on the original surface."""
    dense = naca4("2412", n_panels=400)
    path = tmp_path / "naca2412.dat"
    write_selig(dense, path)
    af = load_selig(path, n_panels=160)
    assert af.n_panels == 160 and signed_area(af.X, af.Y) < 0

    ref = naca4("2412", n_panels=20000)
    d = distance_to_polyline(af.X, af.Y, ref.X, ref.Y)
    assert d.max() < 2e-5, f"max distance to true surface = {d.max():.2e}"
    # The repaneled LE is the true minimum-x point. On a cambered NACA section that is slightly
    # ahead of x = 0, because x_U = x - y_t sin(theta_c) < 0 just behind the nose.
    x = np.linspace(0, 0.01, 2_000_001)
    yc, dyc = naca4_camber(x, 0.02, 0.4)
    true_min = np.min(x - naca4_thickness(x, 0.12) * np.sin(np.arctan(dyc)))
    assert af.X.min() == pytest.approx(true_min, abs=1e-8)


def distance_to_polyline(px, py, X, Y):
    """Shortest distance from each point (px, py) to the polyline through (X, Y) (segments, not nodes)."""
    ax, ay, bx, by = X[:-1], Y[:-1], X[1:], Y[1:]
    dx, dy = bx - ax, by - ay
    t = ((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / (dx**2 + dy**2)
    t = np.clip(t, 0.0, 1.0)
    return np.min(np.hypot(px[:, None] - (ax + t * dx), py[:, None] - (ay + t * dy)), axis=1)


def test_karman_trefftz_reduces_to_joukowski():
    kt = KarmanTrefftz(mu=complex(-0.1, 0.1), tau_deg=0.0)
    zeta = kt.circle(np.linspace(0.1, 6.0, 50))
    np.testing.assert_allclose(kt.z(zeta), zeta + 1 / zeta, rtol=1e-12)


def test_karman_trefftz_derivative():
    kt = KarmanTrefftz()
    zeta = kt.circle(np.linspace(0.3, 6.0, 40))
    h = 1e-6
    fd = (kt.z(zeta + h) - kt.z(zeta - h)) / (2 * h)
    np.testing.assert_allclose(kt.dz_dzeta(zeta), fd, rtol=1e-6)


@pytest.mark.parametrize("tau", [5.0, 10.0, 20.0])
def test_karman_trefftz_geometry(tau):
    af = KarmanTrefftz(mu=complex(-0.08, 0.06), tau_deg=tau).airfoil(n_panels=400)
    assert signed_area(af.X, af.Y) < 0
    # nodes are uniform in circle angle, so none sits exactly on the LE: x/c >= 0, nearest ~ 0
    assert 0.0 <= af.X.min() < 1e-4 and af.X[0] == pytest.approx(1.0)
    assert af.Y[0] == af.Y[-1] == 0.0
    # included TE angle from the first and last panels -> tau as the panels shrink
    t_lower = np.arctan2(af.Y[1] - af.Y[0], af.X[1] - af.X[0])
    t_upper = np.arctan2(af.Y[-2] - af.Y[-1], af.X[-2] - af.X[-1])
    included = np.angle(np.exp(1j * (t_upper - t_lower)))  # wrap to (-pi, pi]
    assert np.rad2deg(abs(included)) == pytest.approx(tau, abs=0.5)
