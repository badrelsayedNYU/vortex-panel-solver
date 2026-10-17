"""Compare against XFOIL inviscid results (spec §5.3).

Reads the owner-generated files in data/xfoil/ (see data/xfoil/README.md) and writes
figures/xfoil_comparison.png and results/xfoil_comparison.md. Skips cleanly, listing what is
missing, if the data are not there yet.
Usage: python scripts/xfoil_compare.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from panel.geometry import naca4
from panel.solver import PanelSolver
from panel.xfoil_io import read_cp, read_polar

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "xfoil"
CODES = ("0012", "2412")
ALPHAS = (0, 4, 8)
N = 160


def expected_files():
    files = [DATA / f"naca{c}_a{a}.cp" for c in CODES for a in ALPHAS]
    return files + [DATA / f"naca{c}_polar.txt" for c in CODES]


def main():
    missing = [f.name for f in expected_files() if not f.exists()]
    if missing:
        print("XFOIL comparison skipped; missing data/xfoil/: " + ", ".join(missing))
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    lines = ["| Airfoil | alpha (deg) | C_l panel (N = 160) | C_l XFOIL | difference | C_m panel | C_m XFOIL |",
             "|---|---|---|---|---|---|---|"]
    for ax, code in zip(axes, CODES):
        solver = PanelSolver(naca4(code, n_panels=N, closed_te=True))
        polar = read_polar(DATA / f"naca{code}_polar.txt")
        for alpha, color in zip(ALPHAS, ("tab:blue", "tab:orange", "tab:red")):
            res = solver.solve(alpha)
            x, cp = read_cp(DATA / f"naca{code}_a{alpha}.cp")
            ax.plot(x, cp, color="black", lw=1)
            ax.plot(res.xc, res.Cp, "o", ms=2.5, mfc="none", color=color, label=fr"panel, $\alpha$ = {alpha}°")
            cl_x, cm_x = polar[round(float(alpha), 3)]
            lines.append(f"| NACA {code} | {alpha} | {res.cl_kj:.4f} | {cl_x:.4f} | "
                         f"{100 * (res.cl_kj - cl_x) / cl_x if cl_x else 0:+.2f}% | {res.cm_c4:.4f} | {cm_x:.4f} |")
        ax.plot([], [], color="black", lw=1, label="XFOIL inviscid")
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$C_p$")
        ax.set_title(f"NACA {code} (closed TE): panel method vs. XFOIL")
        ax.legend(fontsize=8, loc="upper right")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "xfoil_comparison.md").write_text("\n".join(lines) + "\n")
    out = ROOT / "figures" / "xfoil_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
