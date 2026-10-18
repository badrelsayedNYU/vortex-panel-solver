"""Analysis tool (spec §6b tests)."""

import numpy as np
import pandas as pd
import pytest

from panel.analysis import corrected_coefficients, make_airfoil, parse_range, summary, sweep
from panel.boundary_layer import viscous_drag
from panel.cli import main
from panel.compressibility import critical_mach
from panel.geometry import naca4
from panel.solver import PanelSolver


def test_parse_range():
    np.testing.assert_allclose(parse_range("-4:12:2"), [-4, -2, 0, 2, 4, 6, 8, 10, 12])
    np.testing.assert_allclose(parse_range("0:1:0.5"), [0, 0.5, 1])
    np.testing.assert_allclose(parse_range("3"), [3])


def test_sweep_matches_direct_single_alpha_calls():
    af = naca4("2412", 120, closed_te=True)
    df = sweep(af, [-2.0, 3.0, 7.0], machs=[0.0, 0.4], reynolds=[1e6])
    for alpha in (-2.0, 3.0, 7.0):
        direct = PanelSolver(naca4("2412", 120, closed_te=True)).solve(alpha)
        rows = df[df["alpha"] == alpha]
        assert (rows["cl0"] == direct.cl_kj).all() and (rows["cm0"] == direct.cm_c4).all()
        assert (rows["cp_min0"] == direct.Cp.min()).all()
        assert (rows["mcr_kt"] == critical_mach(direct.Cp.min(), "KT")).all()
        assert (rows["cd"] == viscous_drag(direct, 1e6).cd).all()
        m0 = rows[rows["mach"] == 0.0].iloc[0]
        assert m0["cl"] == direct.cl_kj


def test_lu_reuse_matches_fresh_solve_per_alpha():
    af = naca4("4415", 100, closed_te=True)
    s = PanelSolver(af)
    for alpha in np.linspace(-5, 12, 6):
        g = s.solve(alpha).gamma
        fresh = np.linalg.solve(s.An, s.rhs(np.deg2rad(alpha)))
        tol = 10 * np.linalg.cond(s.An) * np.finfo(float).eps * np.abs(fresh).max()
        np.testing.assert_allclose(g, fresh, atol=tol, rtol=0)


def test_kt_corrected_lift_increases_with_mach_and_validity_flag():
    af = naca4("0012", 160, closed_te=True)
    df = sweep(af, [2.0], machs=[0.0, 0.3, 0.6])
    cl = df.sort_values("mach")["cl"].to_numpy()
    assert cl[0] < cl[1] < cl[2]
    mcr = df["mcr_kt"].iloc[0]
    np.testing.assert_array_equal(df["compressibility_valid"], df["mach"] < mcr)


def test_kt_lift_correction_close_to_pg_factor_for_thin_section():
    """For a thin section at small alpha, KT lift ~ PG lift = C_l0 / sqrt(1 - M^2)."""
    s = PanelSolver(naca4("0006", 200, closed_te=True))
    sol = s.solve(1.0)
    cl, _ = corrected_coefficients(s, sol, 0.4)
    assert cl == pytest.approx(sol.cl_kj / np.sqrt(1 - 0.16), rel=0.03)


def test_summary_values():
    df = sweep(naca4("2412", 200, closed_te=True), np.arange(-4, 5, 1.0), reynolds=[1e6])
    s = summary(df)
    assert s["cl_alpha_per_rad"] == pytest.approx(2 * np.pi * 1.1, rel=0.03)
    assert s["alpha_L0_deg"] == pytest.approx(-2.15, abs=0.1)
    assert s["cm0"] == pytest.approx(-0.054, abs=0.005)
    assert "max_L/D_Re=1e+06" in s


def test_cli_smoke(tmp_path):
    out = tmp_path / "res"
    assert main(["--naca", "2412", "--alpha=-2:6:2", "--mach", "0,0.5", "--re", "1e6",
                 "--n", "80", "--out", str(out)]) == 0
    for name in ("sweep.csv", "summary.json", "cp.csv", "geometry.png", "cl_alpha.png", "cm_alpha.png",
                 "mcr_alpha.png", "cp.png", "drag_polar.png", "cd_alpha.png", "ld_alpha.png",
                 "transition.png"):
        assert (out / name).exists(), name
    df = pd.read_csv(out / "sweep.csv")
    assert len(df) == 5 * 2


def test_cli_flight_conditions(tmp_path):
    out = tmp_path / "flight"
    dat = tmp_path / "e387.dat"
    from pathlib import Path
    dat.write_text((Path(__file__).resolve().parent.parent / "data/airfoils/e387.dat").read_text())
    assert main(["--dat", str(dat), "--alpha", "0:4:2", "--flight", "50,1000,0.3",
                 "--n", "80", "--out", str(out)]) == 0
    df = pd.read_csv(out / "sweep.csv")
    assert {"lift_per_span_N_m", "drag_per_span_N_m", "cd"} <= set(df.columns)
    assert df["mach"].iloc[0] == pytest.approx(50.0 / 336.43, abs=5e-4)  # a(1 km) = 336.43 m/s (1976 table)


def test_make_airfoil_requires_exactly_one_source():
    with pytest.raises(ValueError):
        make_airfoil()
