"""Integral boundary-layer drag estimate on the panel solution (spec §6, stretch).

Nondimensional: chord c = 1, V_inf = 1, nu = 1/Re. Each surface is marched from the
stagnation point to the trailing edge using the inviscid edge velocity U_e = |V_t|.

  laminar     Thwaites            (spec §6; White, Viscous Fluid Flow §4-6)
  transition  Michel's criterion  (Cebeci & Bradshaw, Momentum Transfer in Boundary Layers)
  turbulent   Head's entrainment  (Cebeci & Bradshaw; Moran, Theory and Applications of Aerodynamics)
  drag        Squire–Young        (spec §6)

Closure choices not fixed by the spec (flagged for the owner):
  - laminar separation (lambda < -0.09) before Michel transition is treated as forcing
    transition there (a short laminar bubble), and is recorded;
  - the turbulent shape factor starts at H = 1.35 with theta continuous across transition;
  - turbulent separation is declared when H > 2.4; the drag estimate is then unreliable;
  - marching stops at x/c = 0.99, before the inviscid TE stagnation (see viscous_drag).
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

LAMBDA_SEP = -0.09
H_TURB_START = 1.35
H_TURB_SEP = 2.4
X_END = 0.99


# ---------------------------------------------------------------- correlations

def thwaites_H(lam):
    """Laminar shape factor H(lambda), Cebeci–Bradshaw fit to Thwaites' table (White eq. 4-136)."""
    lam = np.asarray(lam, dtype=float)
    pos = 2.61 - 3.75 * lam + 5.24 * lam**2
    neg = 2.088 + 0.0731 / (np.maximum(lam, -0.139) + 0.14)
    return np.where(lam >= 0, pos, neg)


def michel_re_theta(re_x):
    """Michel's criterion: transition when Re_theta > 1.174 (1 + 22400/Re_x) Re_x^0.46."""
    return 1.174 * (1.0 + 22400.0 / re_x) * re_x**0.46


def head_H1_from_H(H):
    """Head's mass-flow shape factor H1(H) (Cebeci & Bradshaw)."""
    H = np.asarray(H, dtype=float)
    return np.where(H <= 1.6, 3.3 + 0.8234 * (H - 1.1) ** -1.287,
                    3.3 + 1.5501 * (H - 0.6778) ** -3.064)


def head_H_from_H1(H1):
    """Inverse of head_H1_from_H (Cebeci & Bradshaw)."""
    H1 = np.asarray(H1, dtype=float)
    d = np.maximum(H1 - 3.3, 1e-9)
    return np.where(H1 >= 5.3, 1.1 + 0.86 * d**-0.777, 0.6778 + 1.1536 * d**-0.326)


def head_entrainment(H1):
    """Entrainment function F(H1) = 0.0306 (H1 - 3)^-0.6169 (Head 1958)."""
    return 0.0306 * np.maximum(H1 - 3.0, 1e-9) ** -0.6169


def ludwieg_tillmann_cf(H, re_theta):
    """Turbulent skin friction C_f = 0.246 * 10^(-0.678 H) * Re_theta^-0.268."""
    return 0.246 * 10.0 ** (-0.678 * H) * np.maximum(re_theta, 1.0) ** -0.268


# ---------------------------------------------------------------- marching

@dataclass
class SurfaceBL:
    s: np.ndarray            # arc length from the stagnation point
    x: np.ndarray            # chordwise position of each station
    Ue: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    x_transition: float      # x/c of transition (nan if none before the TE)
    laminar_separation: bool # transition was forced by laminar separation
    x_separation: float      # x/c of turbulent separation (nan if attached to the TE)
    cd: float                # Squire–Young contribution of this surface


def thwaites(s, Ue, nu):
    """theta^2 = 0.45 nu / U_e^6 * integral_0^s U_e^5 ds;  lambda = theta^2/nu * dU_e/ds  (spec §6).

    The integral is exact for piecewise-linear U_e: over each interval of length h,
    integral U_e^5 ds = h/6 * sum_{j=0..5} U_k^j U_{k+1}^(5-j). (The trapezoid rule is badly wrong
    next to a stagnation point, where U_e^5 rises from zero.)
    If s[0] is a stagnation point (U_e = 0), theta^2 there takes its limit 0.075 nu / (dU_e/ds);
    otherwise (e.g. a flat plate's leading edge) theta = 0 there.
    """
    dUe = np.gradient(Ue, s)
    a, b = Ue[:-1], Ue[1:]
    seg = np.diff(s) / 6.0 * sum(a**j * b ** (5 - j) for j in range(6))
    integral = np.r_[0.0, np.cumsum(seg)]
    theta2 = np.zeros_like(Ue, dtype=float)
    pos = Ue > 0
    theta2[pos] = 0.45 * nu * integral[pos] / Ue[pos] ** 6
    if Ue[0] == 0.0:
        theta2[0] = 0.075 * nu / dUe[0]
    return np.sqrt(theta2), theta2 / nu * dUe, dUe


def march_surface(s, x, Ue, re, x_trip=None):
    """Laminar (Thwaites) -> transition (Michel, laminar separation, or a trip at x >= x_trip,
    whichever comes first) -> turbulent (Head) -> TE."""
    nu = 1.0 / re
    theta_l, lam, dUe = thwaites(s, Ue, nu)
    H_l = thwaites_H(lam)
    re_theta, re_x = Ue * theta_l / nu, np.maximum(Ue * s / nu, 1.0)

    lam_sep = np.where((lam < LAMBDA_SEP) & (s > 0))[0]
    michel = np.where((re_theta > michel_re_theta(re_x)) & (s > 0))[0]
    i_lsep = lam_sep[0] if lam_sep.size else s.size
    i_mich = michel[0] if michel.size else s.size
    tripped = np.where((x >= x_trip) & (s > 0))[0] if x_trip is not None else np.array([], int)
    i_trip = tripped[0] if tripped.size else s.size
    i_tr = min(i_lsep, i_mich, i_trip)

    theta, H = theta_l.copy(), H_l.copy()
    x_tr, x_sep = np.nan, np.nan
    if i_tr < s.size:
        x_tr = x[i_tr]
        dUe_of = lambda q: np.interp(q, s, dUe)
        Ue_of = lambda q: np.interp(q, s, Ue)

        def rhs(q, y):
            th, H1 = y
            ue, due = Ue_of(q), dUe_of(q)
            Hq = head_H_from_H1(H1)
            cf = ludwieg_tillmann_cf(Hq, ue * th / nu)
            dth = 0.5 * cf - (Hq + 2.0) * th / ue * due
            dH1 = (head_entrainment(H1) - H1 * (due * th / ue + dth)) / th
            return [dth, dH1]

        sep = lambda q, y: head_H_from_H1(y[1]) - H_TURB_SEP
        sep.terminal = True
        y0 = [theta_l[i_tr], float(head_H1_from_H(H_TURB_START))]
        sol = solve_ivp(rhs, (s[i_tr], s[-1]), y0, events=sep, dense_output=True,
                        max_step=(s[-1] - s[i_tr]) / 200, rtol=1e-7, atol=1e-12)
        q = s[i_tr:]
        q_end = sol.t[-1]
        y = sol.sol(np.minimum(q, q_end))
        theta[i_tr:], H[i_tr:] = y[0], head_H_from_H1(y[1])
        if sol.t_events[0].size:
            x_sep = float(np.interp(sol.t_events[0][0], s, x))
    elif i_lsep < s.size:
        x_tr = x[i_lsep]

    cd = 2.0 * theta[-1] * Ue[-1] ** ((H[-1] + 5.0) / 2.0)
    return SurfaceBL(s, x, Ue, theta, H, x_tr,
                     bool(i_lsep < s.size and i_lsep <= min(i_mich, i_trip)), x_sep, cd)


@dataclass
class ViscousResult:
    cd: float
    upper: SurfaceBL
    lower: SurfaceBL

    @property
    def separated(self):
        return bool(np.isfinite(self.upper.x_separation) or np.isfinite(self.lower.x_separation))


def split_surfaces(solution):
    """Split the panel solution at the front stagnation point into (s, x, U_e) for the upper and
    lower surfaces, each running from the stagnation point (s = 0, U_e = 0) to the TE."""
    Vt, xc, yc = solution.Vt, solution.xc, solution.yc
    flips = np.where(np.sign(Vt[:-1]) != np.sign(Vt[1:]))[0]
    k = flips[np.argmin(xc[flips])]
    w = Vt[k] / (Vt[k] - Vt[k + 1])
    xs, ys = xc[k] + w * (xc[k + 1] - xc[k]), yc[k] + w * (yc[k + 1] - yc[k])

    def build(idx, ue):
        px, py = np.r_[xs, xc[idx]], np.r_[ys, yc[idx]]
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        return s, px, np.r_[0.0, ue]

    upper = build(np.arange(k + 1, xc.size), Vt[k + 1:])
    lower = build(np.arange(k, -1, -1), -Vt[k::-1])
    return upper, lower


def viscous_drag(solution, re, x_end=X_END):
    """Profile drag C_d = sum over both surfaces of 2 theta_TE (U_e,TE)^((H_TE + 5)/2)  (spec §6).

    Each surface is marched only to x/c = x_end (default 0.99) and Squire–Young is applied there.
    At a sharp, closed TE the inviscid edge velocity drops to a stagnation point in the last ~1% of
    chord (Cp about +0.6 on the last panel); a real boundary layer never sees this because the wake
    removes it, and marching into it triggers spurious turbulent separation. The drag is
    insensitive to the exact cutoff: NACA 0012, Re = 1e6, alpha = 0 gives C_d = 0.0076 / 0.0076 /
    0.0075 for x_end = 0.995 / 0.99 / 0.98.
    """
    (su, xu, uu), (sl, xl, ul) = split_surfaces(solution)
    ku, kl = xu <= x_end, xl <= x_end
    upper = march_surface(su[ku], xu[ku], uu[ku], re)
    lower = march_surface(sl[kl], xl[kl], ul[kl], re)
    return ViscousResult(upper.cd + lower.cd, upper, lower)
