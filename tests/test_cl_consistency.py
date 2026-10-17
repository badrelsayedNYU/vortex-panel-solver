"""C_l from Kutta–Joukowski vs. pressure integration (spec §5.5).

Kutta–Joukowski C_l converges fast (5 digits by N = 160). Pressure-integrated C_l uses one Cp
value per straight panel and converges at first order: the gap halves each time N doubles
(about 0.58% at N = 160, 0.29% at N = 320). The spec's 0.5% criterion is therefore checked
at N = 320, and the first-order convergence of the gap is checked separately.
"""

import numpy as np
import pytest

from panel.geometry import KarmanTrefftz, naca4
from panel.solver import PanelSolver

AIRFOILS = {
    "NACA 0012": lambda N: naca4("0012", N, closed_te=True),
    "NACA 2412": lambda N: naca4("2412", N, closed_te=True),
    "NACA 4415": lambda N: naca4("4415", N, closed_te=True),
    "Kármán–Trefftz": lambda N: KarmanTrefftz(complex(-0.1, 0.1), 10.0).airfoil(N),
}


@pytest.mark.parametrize("name", AIRFOILS)
@pytest.mark.parametrize("alpha", [2.0, 4.0, 8.0])
def test_kj_and_pressure_lift_agree_within_half_percent(name, alpha):
    res = PanelSolver(AIRFOILS[name](320)).solve(alpha)
    assert abs(res.cl_p - res.cl_kj) / abs(res.cl_kj) < 0.005


@pytest.mark.parametrize("name", AIRFOILS)
def test_lift_gap_and_pressure_drag_converge_first_order(name):
    gaps, drags = [], []
    for N in (80, 160, 320, 640):
        res = PanelSolver(AIRFOILS[name](N)).solve(4.0)
        gaps.append(abs(res.cl_p - res.cl_kj))
        drags.append(abs(res.cd_p))
    for series in (gaps, drags):
        ratios = np.array(series[:-1]) / np.array(series[1:])
        assert np.all((ratios > 1.8) & (ratios < 2.2)), f"ratios {ratios}"
