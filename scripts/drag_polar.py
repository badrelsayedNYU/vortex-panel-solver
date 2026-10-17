"""Viscous drag estimate (spec §6): drag polars, C_d vs. alpha and transition locations.

Points where the boundary layer separates before the TE are drawn as open red markers: the
Squire–Young drag is unreliable there. If an XFOIL viscous polar for NACA 0012 at Re = 1e6 is
present (data/xfoil/naca0012_visc_re1e6.txt), it is overlaid.
Writes figures/drag_polar.png and results/drag_polar.csv.
Usage: python scripts/drag_polar.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from panel.boundary_layer import viscous_drag
from panel.geometry import naca4
from panel.solver import PanelSolver
from panel.xfoil_io import read_polar_full

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = np.arange(-4.0, 10.01, 1.0)
N = 200


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    rows = ["airfoil,re,alpha,cl,cd,xtr_upper,xtr_lower,separated"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    styles = {5e5: ":", 1e6: "-", 3e6: "--"}
    for code, color in (("0012", "tab:blue"), ("2412", "tab:red")):
        s = PanelSolver(naca4(code, n_panels=N, closed_te=True))
        sols = [s.solve(a) for a in ALPHAS]
        for re, ls in styles.items():
            res = [viscous_drag(r, re) for r in sols]
            cl = np.array([r.cl_kj for r in sols])
            cd = np.array([v.cd for v in res])
            sep = np.array([v.separated for v in res])
            xu = np.array([v.upper.x_transition for v in res])
            xl = np.array([v.lower.x_transition for v in res])
            label = f"NACA {code}, Re = {re:.0e}"
            axes[0].plot(cd, cl, ls, color=color, label=label)
            axes[1].plot(ALPHAS, cd, ls, color=color, label=label)
            for ax, xs, ys in ((axes[0], cd, cl), (axes[1], ALPHAS, cd)):
                ax.plot(xs[sep], ys[sep], "o", mfc="none", mec="red", ms=6)
            if re == 1e6:
                axes[2].plot(ALPHAS, xu, "-o", ms=3, color=color, label=f"NACA {code} upper")
                axes[2].plot(ALPHAS, xl, "--s", ms=3, color=color, label=f"NACA {code} lower")
            rows += [f"NACA {code},{re:g},{a:g},{c:.5f},{d:.6f},{u:.4f},{lo:.4f},{int(sp)}"
                     for a, c, d, u, lo, sp in zip(ALPHAS, cl, cd, xu, xl, sep)]

    xfoil = ROOT / "data" / "xfoil" / "naca0012_visc_re1e6.txt"
    if xfoil.exists():
        pol = read_polar_full(xfoil)
        axes[0].plot(pol["CD"], pol["CL"], "k.", ms=4, label="XFOIL, NACA 0012, Re = 1e6")
        axes[1].plot(pol["alpha"], pol["CD"], "k.", ms=4, label="XFOIL, NACA 0012, Re = 1e6")
    axes[0].plot([], [], "o", mfc="none", mec="red", label="separated before TE (unreliable)")

    axes[0].set_xlabel("$C_d$")
    axes[0].set_ylabel("$C_l$ (inviscid)")
    axes[0].set_title("Drag polar")
    axes[1].set_xlabel(r"$\alpha$ (deg)")
    axes[1].set_ylabel("$C_d$")
    axes[1].set_title("Profile drag (Squire–Young)")
    axes[2].set_xlabel(r"$\alpha$ (deg)")
    axes[2].set_ylabel("transition $x/c$")
    axes[2].set_title("Transition location, Re = 1e6 (Michel)")
    for ax in axes:
        ax.legend(fontsize=7.5)
    fig.suptitle("Integral boundary layer on the inviscid panel solution: Thwaites -> Michel -> Head")
    fig.savefig(ROOT / "figures" / "drag_polar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "results" / "drag_polar.csv").write_text("\n".join(rows) + "\n")
    print("wrote figures/drag_polar.png, results/drag_polar.csv")


if __name__ == "__main__":
    main()
