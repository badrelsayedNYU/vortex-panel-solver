"""`airfoil-analyze`: sweep an airfoil over alpha x Mach x Re with this repo's panel solver (spec §6b).

Examples:
  airfoil-analyze --naca 2412 --alpha=-4:12:1 --mach 0,0.3,0.5 --re 1e6 --out results/
  airfoil-analyze --dat e387.dat --alpha 0:10:0.5 --flight 50,1000,0.3 --out results/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panel import plots  # noqa: E402
from panel.analysis import cp_distributions, make_airfoil, parse_range, summary, sweep  # noqa: E402
from panel.atmosphere import flight_to_mach_re  # noqa: E402

PHYSICS_NOTE = ("Note: in the inviscid, incompressible solution C_l and Cp do not depend on velocity; "
                "velocity enters only through Mach (compressibility correction) and Reynolds "
                "(boundary layer). The inviscid method does not predict stall.")


def floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def build_parser():
    p = argparse.ArgumentParser(prog="airfoil-analyze", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=PHYSICS_NOTE)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--naca", help="NACA 4-digit code, e.g. 2412")
    src.add_argument("--dat", help="Selig-format coordinate file")
    p.add_argument("--n", type=int, default=160, help="number of panels (default 160)")
    p.add_argument("--alpha", default="-4:12:1", help="start:stop:step in degrees (stop included)")
    cond = p.add_mutually_exclusive_group()
    cond.add_argument("--mach", type=floats, help="comma-separated Mach numbers")
    cond.add_argument("--flight", type=floats, metavar="V,h,c",
                      help="speed [m/s], altitude [m], chord [m] -> Mach and Re (1976 US Std Atm)")
    p.add_argument("--re", type=floats, help="comma-separated Reynolds numbers (enables the boundary layer)")
    p.add_argument("--trip", type=float, help="force transition at this x/c on both surfaces")
    p.add_argument("--cp-alpha", type=floats, help="alphas for Cp plots (default: three across the range)")
    p.add_argument("--out", default="results/", help="output directory")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    airfoil = make_airfoil(args.naca, args.dat, args.n)
    alphas = parse_range(args.alpha)

    flight = None
    if args.flight:
        V, h, c = args.flight
        mach, re, atm = flight_to_mach_re(V, h, c)
        machs, reynolds = [mach], [re]
        flight = dict(V=V, h=h, c=c, mach=mach, re=re, rho=float(atm["rho"]))
    else:
        machs, reynolds = args.mach or [0.0], args.re

    df = sweep(airfoil, alphas, machs, reynolds, x_trip=args.trip)
    if flight:  # dimensional per-unit-span loads (secondary outputs)
        q = 0.5 * flight["rho"] * flight["V"] ** 2
        df["lift_per_span_N_m"] = q * flight["c"] * df["cl"]
        if "cd" in df:
            df["drag_per_span_N_m"] = q * flight["c"] * df["cd"]
    df.to_csv(out / "sweep.csv", index=False)
    summ = summary(df) | ({"flight": flight} if flight else {})
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))

    figures = {"geometry": plots.plot_geometry(airfoil)}
    for name, fn in plots.ALL_PLOTS.items():
        if name in ("drag_polar", "cd_alpha", "ld_alpha", "transition") and "cd" not in df:
            continue
        figures[name] = fn(df)
    cp_alpha = args.cp_alpha or [alphas[0], alphas[len(alphas) // 2], alphas[-1]]
    cpdf = cp_distributions(airfoil, cp_alpha, machs)
    cpdf.to_csv(out / "cp.csv", index=False)
    figures["cp"] = plots.plot_cp(cpdf)
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    print(f"{airfoil.name}: {len(alphas)} alphas x {len(machs)} Mach"
          + (f" x {len(reynolds)} Re" if reynolds else "") + f" -> {out}/")
    for k, v in summ.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
    invalid = int((~df["compressibility_valid"]).sum())
    if invalid:
        print(f"  WARNING: {invalid} rows at or above M_cr (compressibility correction invalid)")
    if "separated" in df and df["separated"].any():
        print(f"  WARNING: {int(df['separated'].sum())} rows separate before the TE (drag unreliable)")
    print("  " + PHYSICS_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
