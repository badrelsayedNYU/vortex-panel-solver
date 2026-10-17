"""Panel convergence study (spec §5.4).

Errors in C_l and Cp_min vs. N = 20 ... 320 (and beyond for the fitted order):
  - Kármán–Trefftz section: against the exact conformal-map solution;
  - NACA 0012 (closed TE): against an N = 2560 panel solution.
Writes figures/convergence.png and results/convergence.csv.
Usage: python scripts/convergence.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.optimize import minimize_scalar

from panel.geometry import KarmanTrefftz, naca4
from panel.solver import PanelSolver
from panel.validation import kt_exact_cl, kt_exact_cp

ROOT = Path(__file__).resolve().parent.parent
N_LIST = np.array([20, 40, 80, 160, 320, 640])
ALPHA = 4.0
KT = KarmanTrefftz(complex(-0.1, 0.1), 10.0)


def observed_order(N, err):
    """Least-squares slope p of log(err) = -p log(N) + c."""
    return -np.polyfit(np.log(N), np.log(err), 1)[0]


def kt_exact_cp_min(kt, alpha):
    """Minimum of the exact surface Cp (dense scan in circle angle, then a bounded refine)."""
    phi = np.linspace(-kt.beta - 1e-3, -kt.beta - 2 * np.pi + 1e-3, 200001)
    i = int(np.argmin(kt_exact_cp(kt, phi, alpha)))
    lo, hi = sorted((phi[max(i - 1, 0)], phi[min(i + 1, phi.size - 1)]))
    return minimize_scalar(lambda p: kt_exact_cp(kt, np.array([p]), alpha)[0],
                           bounds=(lo, hi), method="bounded", options={"xatol": 1e-14}).fun


def study():
    rows = []
    cl_ex, cpmin_ex = kt_exact_cl(KT, ALPHA), kt_exact_cp_min(KT, ALPHA)
    ref = PanelSolver(naca4("0012", 2560, closed_te=True)).solve(ALPHA)
    for N in N_LIST:
        r = PanelSolver(KT.airfoil(int(N))).solve(ALPHA)
        rows.append(("Kármán–Trefftz", N, r.cl_kj, r.Cp.min(), abs(r.cl_kj - cl_ex), abs(r.Cp.min() - cpmin_ex)))
        r = PanelSolver(naca4("0012", int(N), closed_te=True)).solve(ALPHA)
        rows.append(("NACA 0012", N, r.cl_kj, r.Cp.min(), abs(r.cl_kj - ref.cl_kj), abs(r.Cp.min() - ref.Cp.min())))
    return rows, {"Kármán–Trefftz": (cl_ex, cpmin_ex, "exact"),
                  "NACA 0012": (ref.cl_kj, ref.Cp.min(), "N = 2560")}


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows, refs = study()
    with open(ROOT / "results" / "convergence.csv", "w") as fh:
        fh.write("airfoil,N,cl,cp_min,cl_error,cp_min_error\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.8f},{r[3]:.8f},{r[4]:.3e},{r[5]:.3e}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, k, qty in ((axes[0], 4, "$C_l$"), (axes[1], 5, r"$C_{p,\min}$")):
        for name, color, marker in (("Kármán–Trefftz", "tab:blue", "o"), ("NACA 0012", "tab:red", "s")):
            N = np.array([r[1] for r in rows if r[0] == name])
            err = np.array([r[k] for r in rows if r[0] == name])
            p = observed_order(N, err)
            ax.loglog(N, err, marker=marker, color=color, mfc="none",
                      label=f"{name} vs. {refs[name][2]} (order {p:.2f})")
            print(f"{name:15s} {qty:16s} errors {' '.join(f'{e:.2e}' for e in err)}  order {p:.2f}")
        for order, ls in ((1, "--"), (2, ":")):
            ax.loglog(N, err[0] * 0.6 * (N / N[0]) ** -order, "k" + ls, lw=0.8, label=f"slope {order}")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(N_LIST, [str(n) for n in N_LIST])
        ax.set_xlabel("Number of panels $N$")
        ax.set_ylabel(f"error in {qty}")
        ax.set_title(fr"{qty} convergence, $\alpha$ = {ALPHA:g}°")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    out = ROOT / "figures" / "convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
