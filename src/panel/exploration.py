"""Interactive studies using the solver and independent references already in the repo."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panel.analysis import make_airfoil
from panel.field import inside, velocity
from panel.geometry import KarmanTrefftz, naca4
from panel.solver import PanelSolver
from panel.validation import kt_exact_at_panels, kt_exact_cl
from panel.xfoil_io import read_cp, read_polar


def flow_field(airfoil, alpha_deg):
    """Incompressible pressure and streamlines outside the selected airfoil."""
    sol = PanelSolver(airfoil).solve(alpha_deg)
    x = np.linspace(-0.35, 1.45, 120)
    y = np.linspace(-0.5, 0.5, 72)
    X, Y = np.meshgrid(x, y)
    u, v = velocity(airfoil, sol.gamma, alpha_deg, X, Y)
    mask = inside(airfoil, X, Y) | ~np.isfinite(u) | ~np.isfinite(v)
    u, v = np.ma.array(u, mask=mask), np.ma.array(v, mask=mask)
    cp = 1 - u**2 - v**2
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    filled = ax.contourf(X, Y, cp, levels=np.linspace(-3, 1, 33), cmap="RdBu_r", extend="both")
    fig.colorbar(filled, ax=ax, label="Pressure coefficient Cp (dimensionless)")
    ax.streamplot(x, y, u, v, color="0.2", density=1.25, linewidth=0.55, arrowsize=0.7)
    ax.fill(airfoil.X, airfoil.Y, color="0.6", edgecolor="0.15", zorder=5)
    ax.set(xlabel="x/c (chord lengths)", ylabel="y/c (chord lengths)", aspect="equal",
           title=f"{airfoil.name}: incompressible flow at alpha = {alpha_deg:g}°")
    return fig


def convergence(naca=None, dat=None, alpha_deg=4.0):
    """Compare the selected geometry against a 640-panel numerical reference, not an exact solution."""
    ref = PanelSolver(make_airfoil(naca, dat, 640)).solve(alpha_deg)
    rows = []
    for n in (40, 80, 160, 320):
        r = PanelSolver(make_airfoil(naca, dat, n)).solve(alpha_deg)
        rows.append(dict(panels=n, cl=r.cl_kj, cp_min=float(r.Cp.min()),
                         cl_error=abs(r.cl_kj - ref.cl_kj),
                         cp_min_error=abs(r.Cp.min() - ref.Cp.min())))
    return pd.DataFrame(rows)


def convergence_figure(df, alpha_deg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, label in zip(axes, ("cl_error", "cp_min_error"), ("Cl", "minimum Cp")):
        ax.loglog(df.panels, np.maximum(df[key], np.finfo(float).eps), "o-")
        ax.set(xlabel="Number of panels N", ylabel=f"Absolute {label} difference (dimensionless)",
               title=f"{label} vs 640-panel reference, alpha = {alpha_deg:g}°")
        ax.grid(True, which="both", alpha=0.25)
    return fig


def exact_validation(alpha_deg=4.0, n=160):
    kt = KarmanTrefftz(complex(-0.1, 0.1), 10.0)
    af = kt.airfoil(n)
    sol = PanelSolver(af).solve(alpha_deg)
    exact = kt_exact_at_panels(af, alpha_deg)
    cl_exact = kt_exact_cl(kt, alpha_deg)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(sol.xc, exact, "k-", label="Exact conformal-map solution")
    ax.plot(sol.xc, sol.Cp, ".", label=f"Panel solver, N = {n}")
    ax.invert_yaxis()
    ax.set(xlabel="x/c (chord lengths)", ylabel="Pressure coefficient Cp (dimensionless)",
           title=f"Kármán–Trefftz reference, alpha = {alpha_deg:g}°")
    ax.legend()
    return fig, pd.DataFrame([dict(cl_panel=sol.cl_kj, cl_exact=cl_exact,
                                  cl_absolute_error=abs(sol.cl_kj-cl_exact))])


def xfoil_validation(data_dir, code="0012", alpha_deg=4):
    """Use checked-in inviscid XFOIL output, independent of the panel implementation."""
    data_dir = Path(data_dir)
    x, cp = read_cp(data_dir / f"naca{code}_a{alpha_deg}.cp")
    cl_ref, cm_ref = read_polar(data_dir / f"naca{code}_polar.txt")[float(alpha_deg)]
    sol = PanelSolver(naca4(code, 160, closed_te=True)).solve(alpha_deg)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(x, cp, "k-", label="XFOIL 6.99, inviscid")
    ax.plot(sol.xc, sol.Cp, ".", label="Panel solver, N = 160")
    ax.invert_yaxis()
    ax.set(xlabel="x/c (chord lengths)", ylabel="Pressure coefficient Cp (dimensionless)",
           title=f"NACA {code} reference, alpha = {alpha_deg}°")
    ax.legend()
    rows = [dict(quantity=q, panel=a, xfoil=b, absolute_difference=abs(a-b))
            for q, a, b in [("Cl", sol.cl_kj, cl_ref), ("Cm about quarter chord", sol.cm_c4, cm_ref)]]
    return fig, pd.DataFrame(rows)
