from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from panel.exploration import convergence, exact_validation, flow_field, xfoil_validation
from panel.geometry import naca4

DATA = Path(__file__).resolve().parent.parent / 'data/xfoil'


def test_flow_field_renders_an_uploaded_or_generated_geometry():
    fig = flow_field(naca4('2412', 80, closed_te=True), 4)
    assert len(fig.axes) == 2  # field plus labelled Cp scale
    assert fig.axes[0].get_xlabel() == 'x/c (chord lengths)'
    assert fig.axes[0].patches  # airfoil is drawn over the masked interior
    plt.close(fig)


def test_refinement_improves_lift_against_the_independent_finer_grid():
    df = convergence(naca='2412')
    assert np.all(np.diff(df.cl_error) < 0)
    assert df.panels.tolist() == [40, 80, 160, 320]
    assert np.isfinite(df.to_numpy()).all()


def test_exact_reference_lift_matches_the_published_accuracy():
    fig, df = exact_validation()
    assert df.cl_absolute_error.iloc[0] / abs(df.cl_exact.iloc[0]) < 0.001
    plt.close(fig)


@pytest.mark.parametrize('code', ['0012', '2412'])
@pytest.mark.parametrize('alpha', [0, 4, 8])
def test_xfoil_reference_comparisons_use_the_checked_in_coefficients(code, alpha):
    fig, df = xfoil_validation(DATA, code, alpha)
    lift = df.iloc[0]
    assert lift.absolute_difference < max(abs(lift.xfoil) * 0.002, 1e-10)
    assert df.iloc[1].absolute_difference < 0.0011
    plt.close(fig)
