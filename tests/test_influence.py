"""Independent check of the closed-form influence coefficients (spec §3.3).

Each coefficient is the velocity induced at control point i by panel j when the vortex
strength varies linearly from 1 at one end to 0 at the other, times 2 pi (because
gamma' = gamma / (2 pi V_inf)). Here that velocity is computed by direct numerical
integration of point vortices along the panel, with no use of the A..Q formulas:

  clockwise point vortex of unit strength at (xs, ys): u = (y - ys)/(2 pi r^2), v = -(x - xs)/(2 pi r^2)
  Cn1_ij = integral_0^S (1 - s/S) [u, v].n_i ds * 2 pi,   Cn2_ij = same with weight s/S
  Ct1, Ct2: same with t_i in place of n_i;   n_i = (-sin th_i, cos th_i), t_i = (cos th_i, sin th_i)
"""

import numpy as np
import pytest
from scipy.integrate import quad

from panel.geometry import naca4
from panel.influence import influence_matrices, panel_coefficients


def quadrature_coefficients(af):
    S, th, xc, yc = af.panels()
    N = S.size
    out = np.zeros((4, N, N))
    for i in range(N):
        n = np.array([-np.sin(th[i]), np.cos(th[i])])
        t = np.array([np.cos(th[i]), np.sin(th[i])])
        for j in range(N):
            if i == j:
                continue

            def vel(s, j=j):
                xs = af.X[j] + s * np.cos(th[j])
                ys = af.Y[j] + s * np.sin(th[j])
                r2 = (xc[i] - xs) ** 2 + (yc[i] - ys) ** 2
                return np.array([(yc[i] - ys) / r2, -(xc[i] - xs) / r2])  # 2 pi * velocity

            for k, (w, d) in enumerate([(lambda s: 1 - s / S[j], n), (lambda s: s / S[j], n),
                                        (lambda s: 1 - s / S[j], t), (lambda s: s / S[j], t)]):
                out[k, i, j] = quad(lambda s: w(s) * vel(s) @ d, 0, S[j], epsabs=1e-13, epsrel=1e-11)[0]
    return out


@pytest.mark.parametrize("code,N", [("2412", 16), ("0012", 12)])
def test_closed_form_matches_quadrature(code, N):
    af = naca4(code, n_panels=N)
    S, th, xc, yc = af.panels()
    closed = np.array(panel_coefficients(xc, yc, af.X[:-1], af.Y[:-1], S, th, th))
    numeric = quadrature_coefficients(af)
    off = ~np.eye(N, dtype=bool)
    for k, name in enumerate(["Cn1", "Cn2", "Ct1", "Ct2"]):
        np.testing.assert_allclose(closed[k][off], numeric[k][off], rtol=1e-8, atol=1e-10,
                                   err_msg=name)


def test_self_influence_principal_value():
    """For a straight panel the normal self-influence is a principal value:
    PV integral of -(1 - s/S)/(S/2 - s) over [0, S] = -1 and of -(s/S)/(S/2 - s) = +1, matching
    the spec's Cn1 = -1, Cn2 = 1 (independent of S)."""
    S = 0.37
    eps = 1e-9
    f1 = lambda s: -(1 - s / S) / (S / 2 - s)
    f2 = lambda s: -(s / S) / (S / 2 - s)
    for f, expect in ((f1, -1.0), (f2, 1.0)):
        pv = quad(f, 0, S / 2 - eps, limit=200)[0] + quad(f, S / 2 + eps, S, limit=200)[0]
        assert pv == pytest.approx(expect, abs=1e-6)


def test_assembly_shapes_and_kutta_row():
    af = naca4("2412", n_panels=20)
    An, At = influence_matrices(af)
    assert An.shape == (21, 21) and At.shape == (20, 21)
    np.testing.assert_array_equal(An[20], np.r_[1.0, np.zeros(19), 1.0])
