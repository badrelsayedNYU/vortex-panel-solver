"""Linear-strength vortex panel solve and surface outputs (spec §3.2, §3.4).

The influence matrix depends only on geometry, so PanelSolver LU-factorizes it once and
reuses the factorization for every angle of attack.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from panel.influence import influence_matrices


@dataclass
class Solution:
    alpha_deg: float
    gamma: np.ndarray   # nondimensional nodal strengths gamma'_j = gamma_j/(2 pi V_inf), shape (N+1,)
    xc: np.ndarray      # control points (panel midpoints), shape (N,)
    yc: np.ndarray
    Vt: np.ndarray      # surface tangential velocity V_i/V_inf at control points
    Cp: np.ndarray
    cl_kj: float        # lift coefficient from Kutta–Joukowski
    cl_p: float         # lift coefficient from pressure integration
    cd_p: float         # pressure drag (zero in exact potential flow; a discretization check)
    cm_c4: float        # pitching moment about the quarter chord, positive nose-up


class PanelSolver:
    def __init__(self, airfoil):
        self.airfoil = airfoil
        self.S, self.theta, self.xc, self.yc = airfoil.panels()
        self.An, self.At = influence_matrices(airfoil)
        self.lu = lu_factor(self.An)
        X = airfoil.X
        self.x_le = X.min()
        self.x_te = 0.5 * (X[0] + X[-1])
        self.chord = self.x_te - self.x_le

    def rhs(self, alpha):
        """Flow tangency sin(theta_i - alpha) for i = 1..N, and 0 for the Kutta row (spec §3.2)."""
        return np.append(np.sin(self.theta - alpha), 0.0)

    def solve(self, alpha_deg):
        alpha = np.deg2rad(alpha_deg)
        gamma = lu_solve(self.lu, self.rhs(alpha))
        return self.outputs(alpha_deg, gamma)

    def outputs(self, alpha_deg, gamma):
        """Surface velocity, Cp, C_l (two ways), C_d (pressure) and C_m,c/4  (spec §3.4).

        V_i/V_inf = cos(theta_i - alpha) + sum_j At_ij gamma'_j;   Cp_i = 1 - (V_i/V_inf)^2
        Kutta–Joukowski: Gamma = sum_j (gamma_j + gamma_{j+1})/2 S_j with gamma = 2 pi V_inf gamma',
                         C_l = 2 Gamma/(V_inf c)
        Pressure: panel force dF = -Cp n S with outward normal n = (-sin theta, cos theta)
                  (clockwise ordering); C_n = sum dF_y / c, C_a = sum dF_x / c,
                  C_l = C_n cos(alpha) - C_a sin(alpha),  C_d = C_a cos(alpha) + C_n sin(alpha)
        Moment about (x_le + c/4, 0), positive nose-up:  C_m = -sum[(x - x_ref) dF_y - y dF_x] / c^2
        """
        alpha = np.deg2rad(alpha_deg)
        th, S, c = self.theta, self.S, self.chord
        Vt = np.cos(th - alpha) + self.At @ gamma
        Cp = 1.0 - Vt**2

        circulation = 2 * np.pi * np.sum(0.5 * (gamma[:-1] + gamma[1:]) * S)
        cl_kj = 2.0 * circulation / c

        dFx = -Cp * (-np.sin(th)) * S
        dFy = -Cp * np.cos(th) * S
        cn, ca = dFy.sum() / c, dFx.sum() / c
        cl_p = cn * np.cos(alpha) - ca * np.sin(alpha)
        cd_p = ca * np.cos(alpha) + cn * np.sin(alpha)
        x_ref = self.x_le + 0.25 * c
        cm = -np.sum((self.xc - x_ref) * dFy - self.yc * dFx) / c**2
        return Solution(alpha_deg, gamma, self.xc, self.yc, Vt, Cp, cl_kj, cl_p, cd_p, cm)


def solve(airfoil, alpha_deg):
    """Convenience wrapper: one airfoil, one angle of attack."""
    return PanelSolver(airfoil).solve(alpha_deg)
