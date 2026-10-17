"""Regenerate every figure in figures/ and table in results/.

Usage: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import convergence
import drag_polar
import mcrit
import xfoil_compare
from panel.field import inside, stagnation_point, velocity
from panel.geometry import KarmanTrefftz, load_selig, naca4, read_selig
from panel.solver import PanelSolver
from panel.validation import (kt_exact_at_panels, kt_exact_cl, kt_exact_cp,
                              thin_airfoil_alpha_l0)

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


def fig_cp_basic(N=160):
    """Milestone 2: Cp on NACA 0012 / 2412 at several alpha, and the open-TE artifact."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, code in zip(axes[:2], ("0012", "2412")):
        solver = PanelSolver(naca4(code, n_panels=N, closed_te=True))
        for alpha, color in ((0, "tab:blue"), (4, "tab:orange"), (8, "tab:red")):
            res = solver.solve(alpha)
            ax.plot(res.xc, res.Cp, color=color, lw=1.3,
                    label=fr"$\alpha$ = {alpha}°,  $C_l$ = {res.cl_kj:.3f}")
        ax.set_title(f"NACA {code} (closed TE), N = {N}")

    ax = axes[2]
    for closed, style, label in ((False, "-", "standard open TE (gap not paneled)"),
                                 (True, "--", "closed TE")):
        res = PanelSolver(naca4("0012", n_panels=N, closed_te=closed)).solve(0)
        ax.plot(res.xc, res.Cp, style, lw=1.3, label=label)
    ax.set_title(r"NACA 0012, $\alpha$ = 0°: trailing-edge closure")

    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$C_p$")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Linear-strength vortex panel method: surface pressure (inviscid, incompressible)")
    out = FIG_DIR / "cp_basic.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_kt_validation():
    """Milestone 3 (§5.1): panel Cp vs. the exact Kármán–Trefftz solution."""
    kt = KarmanTrefftz(complex(-0.1, 0.1), 10.0)
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")

    phi = np.linspace(-kt.beta - 1e-4, -kt.beta - 2 * np.pi + 1e-4, 4000)  # TE excluded (0/0)
    x_le, chord = kt.chord_transform()
    x_exact = (kt.z(kt.circle(phi)).real - x_le) / chord
    af80 = kt.airfoil(80)
    s80 = PanelSolver(af80)
    for alpha, color in ((0, "tab:blue"), (4, "tab:orange"), (8, "tab:red")):
        ax.plot(x_exact, kt_exact_cp(kt, phi, alpha), color="black", lw=1)
        res = s80.solve(alpha)
        ax.plot(res.xc, res.Cp, "o", ms=3, mfc="none", color=color,
                label=fr"$\alpha$ = {alpha}°: $C_l$ = {res.cl_kj:.4f} (exact {kt_exact_cl(kt, alpha):.4f})")
    ax.plot([], [], color="black", lw=1, label="exact (conformal map)")
    ax.invert_yaxis()
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.set_title(r"Kármán–Trefftz section ($\mu$ = (-0.1, 0.1), $\tau$ = 10°), N = 80")
    ax.legend(fontsize=8, loc="lower right")

    for N, color in ((80, "tab:gray"), (160, "tab:blue"), (320, "tab:green"), (640, "tab:red")):
        af = kt.airfoil(N)
        res = PanelSolver(af).solve(4)
        err = np.abs(res.Cp - kt_exact_at_panels(af, 4))
        ax_err.semilogy(res.xc, err, ".", ms=2.5, color=color,
                        label=f"N = {N}: max {err.max():.3f}, max excl. TE panels {err[2:-2].max():.3f}")
    ax_err.set_xlabel("$x/c$")
    ax_err.set_ylabel(r"$|C_p - C_{p,\mathrm{exact}}|$")
    ax_err.set_title(r"Pointwise error, $\alpha$ = 4°")
    ax_err.legend(fontsize=7.5, loc="upper center")
    out = FIG_DIR / "kt_validation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_thin_airfoil():
    """Milestone 3 (§5.2): C_l(alpha) vs. thin-airfoil theory, and the slope vs. thickness."""
    fig, (ax, ax_slope) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    alphas = np.arange(-6, 12.1, 1.0)
    for code, color in (("0006", "tab:blue"), ("0012", "tab:orange"), ("0018", "tab:green"),
                        ("2412", "tab:red")):
        s = PanelSolver(naca4(code, n_panels=320, closed_te=True))
        ax.plot(alphas, [s.solve(a).cl_kj for a in alphas], "o-", ms=3, color=color,
                label=f"NACA {code}, panel")
    a_l0 = np.rad2deg(thin_airfoil_alpha_l0(0.02, 0.4))
    ax.plot(alphas, 2 * np.pi * np.deg2rad(alphas), "k--", lw=1, label=r"thin airfoil, symmetric: $2\pi\alpha$")
    ax.plot(alphas, 2 * np.pi * np.deg2rad(alphas - a_l0), "k:", lw=1.2,
            label=fr"thin airfoil, NACA 2412 ($\alpha_{{L0}}$ = {a_l0:.2f}°)")
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel("$C_l$")
    ax.set_title("Lift curves, N = 320 (inviscid: no stall)")
    ax.legend(fontsize=8)

    t = np.array([0.06, 0.09, 0.12, 0.15, 0.18])
    slopes = []
    for ti in t:
        s = PanelSolver(naca4(f"00{int(round(100 * ti)):02d}", n_panels=320, closed_te=True))
        cl = [s.solve(a).cl_kj for a in (-2.0, 0.0, 2.0)]
        slopes.append(np.rad2deg(np.polyfit([-2.0, 0.0, 2.0], cl, 1)[0]) / (2 * np.pi))
    k, b = np.polyfit(t, slopes, 1)
    tt = np.linspace(0, 0.2, 50)
    ax_slope.plot(t, slopes, "o", color="tab:blue", label="panel method, NACA 00xx")
    ax_slope.plot(tt, k * tt + b, "--", color="tab:blue", lw=1,
                  label=fr"linear fit: $C_{{l,\alpha}}/2\pi$ = {b:.3f} + {k:.2f} $t/c$")
    ax_slope.axhline(1.0, color="black", lw=1, label=r"thin airfoil: $2\pi$")
    ax_slope.set_xlabel("thickness ratio $t/c$")
    ax_slope.set_ylabel(r"$C_{l,\alpha}\,/\,2\pi$")
    ax_slope.set_title("Lift slope tends to $2\\pi$ as thickness vanishes")
    ax_slope.legend(fontsize=8)
    out = FIG_DIR / "thin_airfoil.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_streamlines(code="2412", alpha=8.0, N=200):
    """Milestone 5: streamlines and the pressure field, with the stagnation point marked."""
    af = naca4(code, n_panels=N, closed_te=True)
    res = PanelSolver(af).solve(alpha)
    xs, ys = stagnation_point(res)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.9, 1]})
    for ax, (x0, x1, y0, y1, n, density) in zip(axes, ((-0.6, 1.8, -0.6, 0.6, 360, 2.2),
                                                       (-0.08, 0.12, -0.08, 0.06, 260, 2.0))):
        X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, int(n * (y1 - y0) / (x1 - x0))))
        U, V = velocity(af, res.gamma, alpha, X, Y)
        mask = inside(af, X, Y)
        U, V = np.ma.array(U, mask=mask), np.ma.array(V, mask=mask)
        cp = 1.0 - (U**2 + V**2)
        pc = ax.contourf(X, Y, cp, levels=np.linspace(-2.5, 1.0, 36), cmap="RdBu_r", extend="both")
        ax.streamplot(X, Y, U, V, density=density, color="k", linewidth=0.6, arrowsize=0.6)
        ax.fill(af.X, af.Y, color="0.75", zorder=3)
        ax.plot(af.X, af.Y, color="k", lw=0.8, zorder=4)
        ax.plot(xs, ys, "o", color="gold", mec="k", ms=7, zorder=5, label="stagnation point")
        ax.plot(af.X[0], af.Y[0], "s", color="gold", mec="k", ms=6, zorder=5,
                label="trailing edge (Kutta: smooth outflow)")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        ax.grid(False)
    fig.colorbar(pc, ax=axes, label="$C_p$", shrink=0.9)
    axes[0].set_title(fr"NACA {code}, $\alpha$ = {alpha:g}°, $C_l$ = {res.cl_kj:.3f} (inviscid, N = {N})")
    axes[1].set_title(f"Leading edge: stagnation at x/c = {xs:.4f}, y/c = {ys:.4f}")
    axes[1].set_xticks([-0.05, 0.0, 0.05, 0.1])
    axes[0].legend(loc="lower right", fontsize=8)
    out = FIG_DIR / "streamlines.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    fig_geometry()
    fig_cp_basic()
    fig_kt_validation()
    fig_thin_airfoil()
    fig_streamlines()
    convergence.main()
    mcrit.main()
    drag_polar.main()
    xfoil_compare.main()


if __name__ == "__main__":
    main()
