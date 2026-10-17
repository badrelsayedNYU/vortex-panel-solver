"""Regenerate the validation figures in figures/.

Usage: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from panel.geometry import KarmanTrefftz, load_selig, naca4, read_selig

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fig_geometry():
    """Milestone 1: NACA sections, TE closure option, .dat repaneling, Kármán–Trefftz."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")

    ax = axes[0, 0]
    for code, color in (("0012", "tab:blue"), ("2412", "tab:orange"), ("4415", "tab:green")):
        af = naca4(code, n_panels=40)
        ax.plot(af.X, af.Y, "-o", ms=2.5, lw=1, color=color, label=f"NACA {code}")
    ax.plot(af.X[0], af.Y[0], "kv", ms=7, label="node 1 (lower TE)")
    ax.annotate("clockwise", xy=(0.45, -0.09), fontsize=8)
    ax.set_title("NACA 4-digit, N = 40 panels, cosine spacing")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    for closed, style, label in ((False, "-o", "standard ($-0.1015x^4$, open TE)"),
                                 (True, "--s", "closed TE ($-0.1036x^4$)")):
        af = naca4("0012", n_panels=400, closed_te=closed)
        ax.plot(af.X, af.Y, style, ms=3, mfc="none", lw=1, label=label)
    ax.set_xlim(0.95, 1.004)
    ax.set_ylim(-0.006, 0.006)
    ax.set_title("NACA 0012 trailing edge (zoom)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    name, x, y = read_selig(ROOT / "data" / "airfoils" / "e387.dat")
    af = load_selig(ROOT / "data" / "airfoils" / "e387.dat", n_panels=60)
    ax.plot(x, y, "x", ms=4, color="gray", label=f"{name}: {x.size} file points (UIUC)")
    ax.plot(af.X, af.Y, "-o", ms=2.5, lw=1, mfc="none", color="tab:red",
            label="spline repaneled, N = 60")
    ax.set_title("Selig .dat loader + cosine repaneling")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    for kt, color in ((KarmanTrefftz(complex(-0.1, 0.0), 10.0), "tab:blue"),
                      (KarmanTrefftz(complex(-0.1, 0.1), 10.0), "tab:purple"),
                      (KarmanTrefftz(complex(-0.1, 0.1), 0.0), "tab:brown")):
        af = kt.airfoil(n_panels=40)
        label = (r"$\mu$ = (%.2f, %.2f), $\tau$ = %g°" % (kt.mu.real, kt.mu.imag, kt.tau_deg)
                 + (" (Joukowski)" if kt.tau_deg == 0 else ""))
        ax.plot(af.X, af.Y, "-o", ms=2.5, lw=1, color=color, label=label)
    ax.set_title("Kármán–Trefftz sections (exact-solution geometry)")
    ax.legend(fontsize=8, loc="upper right")

    for ax in axes.flat:
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        if ax is not axes[0, 1]:
            ax.set_aspect("equal")
            ax.set_ylim(-0.15, 0.25)
    out = FIG_DIR / "geometry.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    fig_geometry()


if __name__ == "__main__":
    main()
