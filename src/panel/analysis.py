"""Sweeps over alpha x Mach x Re built on this repo's own solver (spec §6b). No external aero code.

The influence matrix depends only on geometry, so one PanelSolver (one LU factorization) serves
every angle of attack; Mach and Reynolds enter only through post-processing:
  - Mach: Kármán–Tsien / Prandtl–Glauert corrections and the critical Mach number;
  - Reynolds: the integral boundary layer on the incompressible edge velocity.
"""

import numpy as np
import pandas as pd

from panel.boundary_layer import viscous_drag
from panel.compressibility import critical_mach, karman_tsien, kt_singular_mach
from panel.geometry import load_selig, naca4
from panel.solver import PanelSolver


def make_airfoil(naca=None, dat=None, n_panels=160):
    """Airfoil from a NACA 4-digit code (closed TE, see README) or a Selig .dat file."""
    if (naca is None) == (dat is None):
        raise ValueError("give exactly one of naca or dat")
    return naca4(naca, n_panels, closed_te=True) if naca is not None else load_selig(dat, n_panels)


def parse_range(text):
    """'start:stop:step' (inclusive of stop) or a single value -> array of floats."""
    parts = [float(p) for p in str(text).split(":")]
    if len(parts) == 1:
        return np.array(parts)
    start, stop, step = parts if len(parts) == 3 else (parts[0], parts[1], 1.0)
    return np.round(np.arange(start, stop + step / 2, step), 10)


def corrected_coefficients(solver, sol, mach):
    """Kármán–Tsien corrected C_l and C_m at Mach M.

    The corrected Cp is integrated with the same pressure integration as the incompressible Cp,
    and the ratio is applied to the Kutta–Joukowski C_l, so the first-order error of pressure
    integration cancels: C_l(M) = C_l,KJ * C_l,p[Cp_KT] / C_l,p[Cp0] (likewise C_m).
    Returns (cl, cm), or (nan, nan) past the KT singularity of the minimum Cp.
    """
    if mach == 0:
        return sol.cl_kj, sol.cm_c4
    if mach >= kt_singular_mach(sol.Cp.min()):
        return np.nan, np.nan
    cp = karman_tsien(sol.Cp, mach)
    alpha = np.deg2rad(sol.alpha_deg)
    th, S, c = solver.theta, solver.S, solver.chord
    dFx, dFy = cp * np.sin(th) * S, -cp * np.cos(th) * S
    cl_p = (dFy.sum() * np.cos(alpha) - dFx.sum() * np.sin(alpha)) / c
    cm = -np.sum((solver.xc - (solver.x_le + 0.25 * c)) * dFy - solver.yc * dFx) / c**2
    cl = sol.cl_kj * cl_p / sol.cl_p if sol.cl_p != 0 else 0.0
    return cl, cm


def sweep(airfoil, alphas, machs=(0.0,), reynolds=None, x_trip=None):
    """One row per alpha x M (x Re): incompressible and corrected coefficients, M_cr with a validity
    flag, and (if reynolds is given) the viscous drag, transition, separation flag and L/D."""
    solver = PanelSolver(airfoil)  # one LU factorization for the whole sweep
    rows = []
    for alpha in alphas:
        sol = solver.solve(alpha)
        cp0_min = sol.Cp.min()
        mcr_kt, mcr_pg = critical_mach(cp0_min, "KT"), critical_mach(cp0_min, "PG")
        visc = {re: viscous_drag(sol, re, x_trip=x_trip) for re in (reynolds or [])}
        for mach in machs:
            cl, cm = corrected_coefficients(solver, sol, mach)
            base = dict(alpha=float(alpha), mach=float(mach), cl0=sol.cl_kj, cm0=sol.cm_c4,
                        cl=cl, cm=cm, cp_min0=cp0_min, mcr_kt=mcr_kt, mcr_pg=mcr_pg,
                        compressibility_valid=bool(mach < mcr_kt))
            if not reynolds:
                rows.append(base)
                continue
            for re in reynolds:
                v = visc[re]
                rows.append(base | dict(re=float(re), cd=v.cd, l_over_d=cl / v.cd,
                                        xtr_upper=v.upper.x_transition, xtr_lower=v.lower.x_transition,
                                        xsep_upper=v.upper.x_separation, xsep_lower=v.lower.x_separation,
                                        separated=v.separated))
    return pd.DataFrame(rows)


def cp_distributions(airfoil, alphas, machs=(0.0,)):
    """Long table of surface Cp (x, cp, alpha, mach) for Cp plots; KT-corrected for M > 0."""
    solver = PanelSolver(airfoil)
    frames = []
    for alpha in alphas:
        sol = solver.solve(alpha)
        for mach in machs:
            if mach > 0 and mach >= kt_singular_mach(sol.Cp.min()):
                continue
            cp = sol.Cp if mach == 0 else karman_tsien(sol.Cp, mach)
            frames.append(pd.DataFrame(dict(x=sol.xc, y=sol.yc, cp=cp, alpha=float(alpha), mach=float(mach))))
    return pd.concat(frames, ignore_index=True)


def summary(df):
    """C_l,alpha (per rad and per deg), alpha_L0 and C_m0 from the incompressible lift line, plus
    the best L/D and its alpha when viscous results are present (spec §6b)."""
    first = df[df["mach"] == df["mach"].min()].drop_duplicates("alpha")
    a, cl, cm = first["alpha"].to_numpy(), first["cl0"].to_numpy(), first["cm0"].to_numpy()
    slope_deg, icpt = np.polyfit(a, cl, 1) if a.size > 1 else (np.nan, np.nan)
    alpha_l0 = -icpt / slope_deg if a.size > 1 else np.nan
    out = {"cl_alpha_per_rad": np.rad2deg(slope_deg), "cl_alpha_per_deg": slope_deg,
           "alpha_L0_deg": alpha_l0, "cm0": float(np.interp(alpha_l0, a, cm)) if a.size > 1 else cm[0]}
    if "cd" in df:
        for re, g in df[df["mach"] == df["mach"].min()].groupby("re"):
            g = g[~g["separated"]]
            if len(g):
                best = g.loc[g["l_over_d"].idxmax()]
                out[f"max_L/D_Re={re:.3g}"] = best["l_over_d"]
                out[f"alpha_max_L/D_Re={re:.3g}"] = best["alpha"]
    return out
