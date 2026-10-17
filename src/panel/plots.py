"""Performance plots from a sweep DataFrame (spec §6b). Each function returns a Matplotlib figure.

Validity flags are drawn, not hidden: points at or above the critical Mach are greyed with an
'x', and viscous points with boundary-layer separation before the TE are open red circles.
"""

import matplotlib.pyplot as plt
import numpy as np

NO_STALL = "Inviscid method: no stall; $C_l$ keeps rising with $\\alpha$."
INVALID_STYLE = dict(marker="x", ls="none", color="0.6", ms=6)
SEP_STYLE = dict(marker="o", ls="none", mfc="none", mec="red", ms=7)


def _fig(title, xlabel, ylabel, figsize=(6.4, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _by_mach(df):
    base = df.drop_duplicates(["alpha", "mach"])
    return [(m, g.sort_values("alpha")) for m, g in base.groupby("mach")]


def plot_geometry(airfoil):
    fig, ax = _fig(f"{airfoil.name}: {airfoil.n_panels} panels", "$x/c$", "$y/c$", (7.5, 3.2))
    ax.plot(airfoil.X, airfoil.Y, "-o", ms=2, lw=0.8)
    ax.set_aspect("equal")
    return fig


def plot_cl_alpha(df):
    fig, ax = _fig("Lift curve (Kármán–Tsien corrected)", r"$\alpha$ (deg)", "$C_l$")
    for m, g in _by_mach(df):
        ok = g["compressibility_valid"].to_numpy()
        line, = ax.plot(g["alpha"][ok], g["cl"][ok], "-o", ms=3, label=f"M = {m:g}")
        ax.plot(g["alpha"][~ok], g["cl"][~ok], **INVALID_STYLE)
    ax.plot([], [], **INVALID_STYLE, label=r"$M_\infty \geq M_{cr}$ (correction invalid)")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.98, NO_STALL, transform=ax.transAxes, va="top", fontsize=7.5, color="0.35")
    return fig


def plot_cm_alpha(df):
    fig, ax = _fig("Pitching moment about c/4", r"$\alpha$ (deg)", r"$C_{m,c/4}$")
    for m, g in _by_mach(df):
        ok = g["compressibility_valid"].to_numpy()
        ax.plot(g["alpha"][ok], g["cm"][ok], "-o", ms=3, label=f"M = {m:g}")
        ax.plot(g["alpha"][~ok], g["cm"][~ok], **INVALID_STYLE)
    ax.legend(fontsize=8)
    return fig


def plot_cp(cpdf):
    fig, ax = _fig("Surface pressure", "$x/c$", "$C_p$")
    for (a, m), g in cpdf.groupby(["alpha", "mach"]):
        ax.plot(g["x"], g["cp"], lw=1.2, label=fr"$\alpha$ = {a:g}°, M = {m:g}")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    return fig


def plot_mcr_alpha(df):
    fig, ax = _fig("Critical Mach number", r"$\alpha$ (deg)", r"$M_{cr}$")
    g = df.drop_duplicates("alpha").sort_values("alpha")
    ax.plot(g["alpha"], g["mcr_kt"], "-o", ms=3, label="Kármán–Tsien")
    ax.plot(g["alpha"], g["mcr_pg"], "--", label="Prandtl–Glauert")
    for m in sorted(df["mach"].unique()):
        if m > 0:
            ax.axhline(m, color="0.6", lw=0.8, ls=":")
            ax.text(g["alpha"].max(), m, f" M = {m:g}", va="bottom", ha="right", fontsize=7.5, color="0.4")
    ax.legend(fontsize=8)
    return fig


def _viscous(df):
    if "cd" not in df:
        raise ValueError("no viscous results: run the sweep with Reynolds numbers")
    return [(re, g.drop_duplicates("alpha").sort_values("alpha"))
            for re, g in df[df["mach"] == df["mach"].min()].groupby("re")]


def _visc_plot(df, x, y, title, xlabel, ylabel):
    fig, ax = _fig(title, xlabel, ylabel)
    for re, g in _viscous(df):
        sep = g["separated"].to_numpy()
        ax.plot(g[x], g[y], "-o", ms=3, label=f"Re = {re:.3g}")
        ax.plot(g[x][sep], g[y][sep], **SEP_STYLE)
    ax.plot([], [], **SEP_STYLE, label="separates before TE (unreliable)")
    ax.legend(fontsize=8)
    return fig


def plot_drag_polar(df):
    return _visc_plot(df, "cd", "cl0", "Drag polar", "$C_d$", "$C_l$")


def plot_cd_alpha(df):
    return _visc_plot(df, "alpha", "cd", "Profile drag (Squire–Young)", r"$\alpha$ (deg)", "$C_d$")


def plot_ld_alpha(df):
    return _visc_plot(df, "alpha", "l_over_d", "Lift-to-drag ratio", r"$\alpha$ (deg)", "$L/D$")


def plot_transition(df):
    fig, ax = _fig("Transition and separation", r"$\alpha$ (deg)", "$x/c$")
    for re, g in _viscous(df):
        line, = ax.plot(g["alpha"], g["xtr_upper"], "-o", ms=3, label=f"transition, upper, Re = {re:.3g}")
        ax.plot(g["alpha"], g["xtr_lower"], "--s", ms=3, color=line.get_color(), label=f"transition, lower, Re = {re:.3g}")
        sep = np.fmin(g["xsep_upper"].to_numpy(), g["xsep_lower"].to_numpy())
        ax.plot(g["alpha"], sep, "v", color="red", ms=6)
    ax.plot([], [], "v", color="red", label="turbulent separation")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    return fig


ALL_PLOTS = {
    "cl_alpha": plot_cl_alpha,
    "cm_alpha": plot_cm_alpha,
    "mcr_alpha": plot_mcr_alpha,
    "drag_polar": plot_drag_polar,
    "cd_alpha": plot_cd_alpha,
    "ld_alpha": plot_ld_alpha,
    "transition": plot_transition,
}
