"""Thin-airfoil theory checks (spec §5.2).

The panel method includes thickness, which raises the lift slope above 2 pi roughly in
proportion to t/c; the slope must therefore tend to 2 pi as t/c -> 0, not equal it.
"""

import numpy as np
import pytest

from panel.geometry import naca4
from panel.solver import PanelSolver
from panel.validation import thin_airfoil_alpha_l0, thin_airfoil_cm_c4


def lift_line(code, alphas=(-2.0, 0.0, 2.0), N=320):
    s = PanelSolver(naca4(code, n_panels=N, closed_te=True))
    cl = [s.solve(a).cl_kj for a in alphas]
    slope_per_deg, intercept = np.polyfit(alphas, cl, 1)
    return np.rad2deg(1.0) * slope_per_deg, -intercept / slope_per_deg, s


def test_lift_slope_tends_to_2pi_with_thickness():
    t = np.array([0.06, 0.12, 0.18])
    slopes = np.array([lift_line(f"00{int(round(100 * ti)):02d}")[0] for ti in t])
    assert np.all(np.diff(slopes) > 0), "thicker sections lift more per radian"
    assert np.all(slopes > 2 * np.pi)
    extrapolated = np.polyval(np.polyfit(t, slopes, 1), 0.0)
    assert extrapolated == pytest.approx(2 * np.pi, rel=0.01)


def test_symmetric_zero_lift_angle_is_zero():
    assert lift_line("0012")[1] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("code", ["2412", "4412"])
def test_cambered_zero_lift_angle_and_moment(code):
    m, p = int(code[0]) / 100, int(code[1]) / 10
    _, alpha_l0, solver = lift_line(code, alphas=(-4.0, -2.0, 0.0, 2.0))
    assert alpha_l0 == pytest.approx(np.rad2deg(thin_airfoil_alpha_l0(m, p)), abs=0.15)
    assert solver.solve(0.0).cm_c4 == pytest.approx(thin_airfoil_cm_c4(m, p), abs=0.006)


def test_thin_airfoil_reference_values():
    """Classical NACA 2412 thin-airfoil values: alpha_L0 = -2.08 deg, C_m,c/4 = -0.053."""
    assert np.rad2deg(thin_airfoil_alpha_l0(0.02, 0.4)) == pytest.approx(-2.08, abs=0.01)
    assert thin_airfoil_cm_c4(0.02, 0.4) == pytest.approx(-0.053, abs=0.001)
