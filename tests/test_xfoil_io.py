"""XFOIL file readers (format check on small synthetic files in XFOIL's layout)."""

import numpy as np

from panel.xfoil_io import read_cp, read_polar


def test_read_cp_new_and_old_formats(tmp_path):
    new = tmp_path / "new.cp"
    new.write_text("# NACA 0012\n# alfa = 4.000\n#    x          y          Cp  \n"
                   "  1.00000   0.00000   0.25000\n  0.50000   0.05000  -0.30000\n")
    old = tmp_path / "old.cp"
    old.write_text("     x          Cp  \n  1.00000   0.25000\n  0.50000  -0.30000\n")
    for path in (new, old):
        x, cp = read_cp(path)
        np.testing.assert_allclose(x, [1.0, 0.5])
        np.testing.assert_allclose(cp, [0.25, -0.30])


def test_read_polar(tmp_path):
    path = tmp_path / "polar.txt"
    path.write_text(
        "       XFOIL         Version 6.99\n\n Calculated polar for: NACA 0012\n\n"
        "  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n"
        " ------- -------- --------- --------- -------- -------- --------\n"
        "   0.000   0.0000   0.00000   0.00000   0.0000   1.0000   1.0000\n"
        "   4.000   0.4836   0.00000   0.00000  -0.0050   1.0000   1.0000\n")
    polar = read_polar(path)
    assert polar[4.0] == (0.4836, -0.0050)
    assert polar[0.0] == (0.0, 0.0)
