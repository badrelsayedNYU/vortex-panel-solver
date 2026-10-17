"""Critical Mach studies (spec §4): M_cr vs. alpha and vs. thickness, Prandtl–Glauert vs. Kármán–Tsien.

Cp0_min is taken from the incompressible panel solution at N = 320 (Cp_min converges at first
order; at N = 320 its error is about 0.006, see figures/convergence.png).
Writes figures/mcrit.png and results/mcrit.csv.
Usage: python scripts/mcrit.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from panel.compressibility import (cp_critical, critical_mach, karman_tsien, kt_singular_mach,
                                   prandtl_glauert)
from panel.geometry import naca4
from panel.solver import PanelSolver

ROOT = Path(__file__).resolve().parent.parent
N = 320


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows = ["study,airfoil,alpha_deg,cp0_min,mcr_pg,mcr_kt"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    ax = axes[0]
    alphas = np.arange(-2.0, 10.01, 0.5)
    for code, color in (("0012", "tab:blue"), ("2412", "tab:red")):
        s = PanelSolver(naca4(code, n_panels=N, closed_te=True))
        cp0 = np.array([s.solve(a).Cp.min() for a in alphas])
        pg = [critical_mach(c, "PG") for c in cp0]
        kt = [critical_mach(c, "KT") for c in cp0]
        ax.plot(alphas, pg, "--", color=color, label=f"NACA {code}, Prandtl–Glauert")
        ax.plot(alphas, kt, "-", color=color, label=f"NACA {code}, Kármán–Tsien")
        rows += [f"alpha,NACA {code},{a:g},{c:.5f},{p:.5f},{k:.5f}" for a, c, p, k in zip(alphas, cp0, pg, kt)]
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$M_{cr}$")
    ax.set_title(r"Critical Mach vs. angle of attack")
    ax.legend(fontsize=8)

    ax = axes[1]
    codes = ["0006", "0008", "0010", "0012", "0015", "0018"]
    t = np.array([int(c[2:]) / 100 for c in codes])
    for alpha, color in ((0.0, "tab:blue"), (2.0, "tab:orange"), (4.0, "tab:red")):
        cp0 = np.array([PanelSolver(naca4(c, n_panels=N, closed_te=True)).solve(alpha).Cp.min() for c in codes])
        pg = [critical_mach(c, "PG") for c in cp0]
        kt = [critical_mach(c, "KT") for c in cp0]
        ax.plot(t, pg, "--o", ms=3, color=color, label=fr"$\alpha$ = {alpha:g}°, PG")
        ax.plot(t, kt, "-s", ms=3, color=color, label=fr"$\alpha$ = {alpha:g}°, KT")
        rows += [f"thickness,NACA {c},{alpha:g},{q:.5f},{p:.5f},{k:.5f}" for c, q, p, k in zip(codes, cp0, pg, kt)]
    ax.set_xlabel("thickness ratio $t/c$ (NACA 00xx)")
    ax.set_ylabel(r"$M_{cr}$")
    ax.set_title("Critical Mach vs. thickness")
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[2]
    M = np.linspace(0.05, 0.999, 400)
    ax.plot(M, cp_critical(M), color="black", lw=1.5, label=r"$C_{p,cr}(M_\infty)$ (sonic)")
    cp0_min = PanelSolver(naca4("0012", n_panels=N, closed_te=True)).solve(0.0).Cp.min()
    ax.plot(M, prandtl_glauert(cp0_min, M), "--", color="tab:blue", label=r"$C_{p,\min}$, Prandtl–Glauert")
    Mk = M[M < kt_singular_mach(cp0_min)]  # KT is undefined past its denominator's zero
    ax.plot(Mk, karman_tsien(cp0_min, Mk), "-", color="tab:blue", label=r"$C_{p,\min}$, Kármán–Tsien")
    for rule, marker in (("PG", "o"), ("KT", "s")):
        m = critical_mach(cp0_min, rule)
        ax.plot(m, cp_critical(m), marker, color="tab:red", ms=7, mfc="none",
                label=f"$M_{{cr}}$ ({rule}) = {m:.3f}")
    ax.set_ylim(-3, 0.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"$M_\infty$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(fr"NACA 0012, $\alpha$ = 0°: $C_{{p0,\min}}$ = {cp0_min:.3f}")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Critical Mach number from the incompressible panel solution (N = 320)")
    fig.savefig(ROOT / "figures" / "mcrit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "results" / "mcrit.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(r for r in rows if r.startswith("thickness") and ",0," in r))
    print("wrote figures/mcrit.png, results/mcrit.csv")


if __name__ == "__main__":
    main()
