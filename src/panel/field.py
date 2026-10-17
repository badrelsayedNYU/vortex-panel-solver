"""Velocity field around the airfoil and stagnation points (spec §3.4, milestone 5)."""

import numpy as np
from matplotlib.path import Path

from panel.influence import panel_coefficients


def assemble_nodes(C1, C2):
    """Combine panel-end coefficients into nodal ones (same rule as the surface assembly, §3.3)."""
    M, N = C1.shape
    A = np.zeros((M, N + 1))
    A[:, :N] += C1
    A[:, 1:] += C2
    return A


def velocity(airfoil, gamma, alpha_deg, x, y):
    """Velocity (u, v)/V_inf at field points  (spec §3.4).

    u = cos(alpha) + sum_j Ax_ij gamma'_j,  v = sin(alpha) + sum_j Ay_ij gamma'_j, where Ax/Ay are
    the §3.3 tangential/normal coefficients evaluated with theta_i = 0, so that t_i = (1, 0) and
    n_i = (0, 1). Valid off the surface; points inside the body are meaningless and should be masked.
    """
    alpha = np.deg2rad(alpha_deg)
    S, theta, _, _ = airfoil.panels()
    xf, yf = np.ravel(x).astype(float), np.ravel(y).astype(float)
    u, v = np.empty_like(xf), np.empty_like(xf)
    chunk = 4096  # bounds memory: each coefficient array is (chunk, N)
    for k in range(0, xf.size, chunk):
        sl = slice(k, k + chunk)
        zero = np.zeros_like(xf[sl])
        Cn1, Cn2, Ct1, Ct2 = panel_coefficients(xf[sl], yf[sl], airfoil.X[:-1], airfoil.Y[:-1],
                                                S, zero, theta)
        u[sl] = np.cos(alpha) + assemble_nodes(Ct1, Ct2) @ gamma
        v[sl] = np.sin(alpha) + assemble_nodes(Cn1, Cn2) @ gamma
    return u.reshape(np.shape(x)), v.reshape(np.shape(x))


def inside(airfoil, x, y):
    """Boolean mask of points inside the closed polygon through the nodes."""
    path = Path(np.column_stack([airfoil.X, airfoil.Y]))
    pts = np.column_stack([np.ravel(x), np.ravel(y)])
    return path.contains_points(pts).reshape(np.shape(x))


def stagnation_point(solution):
    """Front stagnation point: where the surface velocity V_t changes sign near the LE, located by
    linear interpolation between the two control points. Returns (x, y)."""
    Vt, xc, yc = solution.Vt, solution.xc, solution.yc
    flips = np.where(np.sign(Vt[:-1]) != np.sign(Vt[1:]))[0]
    k = flips[np.argmin(xc[flips])]  # the sign change closest to the LE
    w = Vt[k] / (Vt[k] - Vt[k + 1])
    return xc[k] + w * (xc[k + 1] - xc[k]), yc[k] + w * (yc[k + 1] - yc[k])
