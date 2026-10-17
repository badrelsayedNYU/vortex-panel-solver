"""Influence coefficients of linear-strength vortex panels (Kuethe & Chow §5.10; spec §3.3).

Unknowns are the nondimensional nodal strengths gamma'_j = gamma_j / (2 pi V_inf), j = 1..N+1.
Arrays are 0-based here: panel j runs from node j to node j+1.
"""

import numpy as np


def panel_coefficients(xc, yc, X, Y, S, theta_i, theta_j):
    """Cn1, Cn2, Ct1, Ct2 for every (point i, panel j) pair  (spec §3.3; Kuethe & Chow §5.10).

    xc, yc, theta_i: shape (M,), the evaluation points and the direction theta_i that defines
    their normal n_i = (-sin theta_i, cos theta_i) and tangent t_i = (cos theta_i, sin theta_i).
    X, Y: start node of each panel, S: length, theta_j: orientation, shape (N,).
    Returns four (M, N) arrays (i != j formulas; the caller overwrites self-influence):

      A = -(x_i - X_j) cos th_j - (y_i - Y_j) sin th_j       B = (x_i - X_j)^2 + (y_i - Y_j)^2
      C = sin(th_i - th_j)                                    D = cos(th_i - th_j)
      E = (x_i - X_j) sin th_j - (y_i - Y_j) cos th_j
      F = ln(1 + (S_j^2 + 2 A S_j)/B)                         G = atan2(E S_j, B + A S_j)
      P = (x_i - X_j) sin(th_i - 2 th_j) + (y_i - Y_j) cos(th_i - 2 th_j)
      Q = (x_i - X_j) cos(th_i - 2 th_j) - (y_i - Y_j) sin(th_i - 2 th_j)
      Cn2 = D + Q F/(2 S_j) - (A C + D E) G / S_j             Cn1 = D F/2 + C G - Cn2
      Ct2 = C + P F/(2 S_j) + (A D - C E) G / S_j             Ct1 = C F/2 - D G - Ct2
    """
    dx = xc[:, None] - X[None, :]
    dy = yc[:, None] - Y[None, :]
    ti, tj = theta_i[:, None], theta_j[None, :]
    S = S[None, :]

    A = -dx * np.cos(tj) - dy * np.sin(tj)
    B = dx**2 + dy**2
    C = np.sin(ti - tj)
    D = np.cos(ti - tj)
    E = dx * np.sin(tj) - dy * np.cos(tj)
    F = np.log(1.0 + (S**2 + 2.0 * A * S) / B)
    G = np.arctan2(E * S, B + A * S)
    P = dx * np.sin(ti - 2.0 * tj) + dy * np.cos(ti - 2.0 * tj)
    Q = dx * np.cos(ti - 2.0 * tj) - dy * np.sin(ti - 2.0 * tj)

    Cn2 = D + 0.5 * Q * F / S - (A * C + D * E) * G / S
    Cn1 = 0.5 * D * F + C * G - Cn2
    Ct2 = C + 0.5 * P * F / S + (A * D - C * E) * G / S
    Ct1 = 0.5 * C * F - D * G - Ct2
    return Cn1, Cn2, Ct1, Ct2


def influence_matrices(airfoil):
    """Assemble the (N+1) x (N+1) normal matrix An (with the Kutta row) and the N x (N+1)
    tangential matrix At  (spec §3.2-3.3; Kuethe & Chow §5.10).

    Self-influence (i = j): Cn1 = -1, Cn2 = 1, Ct1 = Ct2 = pi/2.
    Assembly (node j is shared by panels j-1 and j):
      An_i,1 = Cn1_i,1;  An_i,j = Cn1_i,j + Cn2_i,j-1 (j = 2..N);  An_i,N+1 = Cn2_i,N
    Kutta row: gamma'_1 + gamma'_{N+1} = 0.
    """
    S, theta, xc, yc = airfoil.panels()
    N = S.size
    Cn1, Cn2, Ct1, Ct2 = panel_coefficients(xc, yc, airfoil.X[:-1], airfoil.Y[:-1], S, theta, theta)
    diag = np.arange(N)
    Cn1[diag, diag], Cn2[diag, diag] = -1.0, 1.0
    Ct1[diag, diag] = Ct2[diag, diag] = 0.5 * np.pi

    An = np.zeros((N + 1, N + 1))
    An[:N, :N] += Cn1
    An[:N, 1:] += Cn2
    An[N, 0] = An[N, N] = 1.0

    At = np.zeros((N, N + 1))
    At[:, :N] += Ct1
    At[:, 1:] += Ct2
    return An, At
