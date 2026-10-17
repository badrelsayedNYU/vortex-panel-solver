"""Reference solutions for validation (spec §5.1-5.2): exact Kármán–Trefftz flow and thin-airfoil theory."""

import numpy as np
from scipy.integrate import quad

from panel.geometry import naca4_camber


# ---------------------------------------------------------------- exact Kármán–Trefftz flow

def kt_circulation(kt, alpha):
    """Kutta condition on the circle: Gamma / V_inf = 4 pi R sin(alpha + beta)  (Katz & Plotkin §6.7).

    Complex potential (clockwise Gamma > 0 gives positive lift):
      W(zeta) = V [(zeta - mu) e^{-i alpha} + R^2 e^{i alpha}/(zeta - mu)] + i Gamma/(2 pi) ln(zeta - mu)
    and dW/dzeta = 0 at the trailing-edge point zeta = b = mu + R e^{-i beta}.
    """
    return 4 * np.pi * kt.R * np.sin(alpha + kt.beta)


def kt_exact_cp(kt, phi, alpha_deg):
    """Exact surface Cp at circle angles phi (not at the TE, where 0/0)  (Katz & Plotkin §6.7).

    dW/dzeta = V [e^{-i alpha} - R^2 e^{i alpha}/(zeta - mu)^2] + i Gamma / (2 pi (zeta - mu))
    |u - iv| in the airfoil plane = |dW/dzeta| / |dz/dzeta|;  Cp = 1 - |u - iv|^2 / V^2.
    The map is a similarity transform to x/c, so Cp is unchanged by the chord normalisation.
    """
    alpha = np.deg2rad(alpha_deg)
    zeta = kt.circle(phi)
    gamma = kt_circulation(kt, alpha)
    dW = np.exp(-1j * alpha) - kt.R**2 * np.exp(1j * alpha) / (zeta - kt.mu) ** 2 \
        + 1j * gamma / (2 * np.pi * (zeta - kt.mu))
    q = np.abs(dW) / np.abs(kt.dz_dzeta(zeta))
    return 1.0 - q**2


def kt_exact_cl(kt, alpha_deg):
    """Exact lift coefficient C_l = 2 Gamma / (V_inf c) = 8 pi R sin(alpha + beta) / c."""
    _, chord = kt.chord_transform()
    return 2 * kt_circulation(kt, np.deg2rad(alpha_deg)) / chord


def kt_exact_at_panels(airfoil, alpha_deg):
    """Exact Cp at the circle angle midway between each panel's end nodes; panel j's control
    point lies within O(h^2) of that surface point."""
    phi = airfoil.meta["phi"]
    return kt_exact_cp(airfoil.meta["kt"], 0.5 * (phi[:-1] + phi[1:]), alpha_deg)


# ---------------------------------------------------------------- thin-airfoil theory

def thin_airfoil_alpha_l0(m, p):
    """Zero-lift angle  alpha_L0 = -(1/pi) integral_0^pi (dy_c/dx)(cos theta0 - 1) d theta0,
    with x = (1 - cos theta0)/2  (Anderson, Fundamentals of Aerodynamics §4.8). Radians."""
    def integrand(t0):
        _, dyc = naca4_camber(np.array([0.5 * (1 - np.cos(t0))]), m, p)
        return dyc[0] * (np.cos(t0) - 1.0)
    split = np.arccos(1 - 2 * p) if 0 < p < 1 else np.pi / 2  # slope kink at x = p
    return -(quad(integrand, 0, split)[0] + quad(integrand, split, np.pi)[0]) / np.pi


def thin_airfoil_cm_c4(m, p):
    """C_m,c/4 = (pi/4)(A2 - A1), A_n = (2/pi) integral_0^pi (dy_c/dx) cos(n theta0) d theta0
    (Anderson §4.8). Independent of alpha."""
    def A(n):
        def integrand(t0):
            _, dyc = naca4_camber(np.array([0.5 * (1 - np.cos(t0))]), m, p)
            return dyc[0] * np.cos(n * t0)
        split = np.arccos(1 - 2 * p) if 0 < p < 1 else np.pi / 2
        return 2 / np.pi * (quad(integrand, 0, split)[0] + quad(integrand, split, np.pi)[0])
    return np.pi / 4 * (A(2) - A(1))
