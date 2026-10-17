"""Validation against the exact Kármán–Trefftz solution (spec §5.1).

Two facts shape these checks:
- C_l converges at second order and is compared directly.
- Cp converges at first order everywhere except on the panels touching the trailing edge.
  At a wedge TE of angle tau, with the Kutta condition satisfied, the exact speed vanishes like
  r^(tau/(2 pi - tau)); for tau = 10 deg the exponent is 0.029, so the exact Cp rises to 1 in a
  layer far thinner than any panel. The TE-panel error therefore stays near 0.05 and is
  reported (not hidden) by the figure script; the convergence check excludes those panels.
"""

import numpy as np
import pytest

from panel.geometry import KarmanTrefftz
from panel.solver import PanelSolver
from panel.validation import kt_exact_at_panels, kt_exact_cl

SECTIONS = [KarmanTrefftz(complex(-0.1, 0.1), 10.0), KarmanTrefftz(complex(-0.08, 0.0), 15.0)]


@pytest.mark.parametrize("kt", SECTIONS, ids=["cambered", "symmetric"])
@pytest.mark.parametrize("alpha", [0.0, 4.0, 8.0])
def test_lift_matches_exact(kt, alpha):
    errors = []
    for N in (80, 160, 320):
        cl = PanelSolver(kt.airfoil(N)).solve(alpha).cl_kj
        errors.append(abs(cl - kt_exact_cl(kt, alpha)))
    if kt_exact_cl(kt, alpha) != 0:
        assert errors[1] / abs(kt_exact_cl(kt, alpha)) < 1e-3   # N = 160: within 0.1%
    if errors[0] > 1e-12:
        assert errors[0] / errors[2] > 9  # ~second order: factor ~16 over two doublings


@pytest.mark.parametrize("alpha", [0.0, 4.0, 8.0])
def test_cp_converges_to_exact_away_from_te(alpha):
    kt = SECTIONS[0]
    max_err = []
    for N in (160, 320, 640):
        af = kt.airfoil(N)
        err = np.abs(PanelSolver(af).solve(alpha).Cp - kt_exact_at_panels(af, alpha))
        max_err.append(err[2:-2].max())  # exclude the two panels touching the TE on each side
    ratios = np.array(max_err[:-1]) / np.array(max_err[1:])
    assert np.all(ratios > 1.7), f"max |dCp| = {max_err}"
    assert max_err[1] < 0.08  # N = 320


def test_te_panel_error_is_bounded():
    kt = SECTIONS[0]
    for N in (160, 320, 640):
        af = kt.airfoil(N)
        err = np.abs(PanelSolver(af).solve(4.0).Cp - kt_exact_at_panels(af, 4.0))
        assert err.max() < 0.07
