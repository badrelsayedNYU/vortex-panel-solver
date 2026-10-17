"""Readers for XFOIL output files (validation data only; spec §5.3). The solver never runs XFOIL."""

from pathlib import Path

import numpy as np


def read_cp(path):
    """Read an XFOIL CPWR file. Header/comment lines are skipped; x is the first column and Cp
    the last (XFOIL 6.99 writes 'x y Cp', older versions 'x Cp'). Returns (x, Cp)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace("#", " # ").split()
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        if len(values) >= 2:
            rows.append((values[0], values[-1]))
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def read_polar(path):
    """Read an XFOIL polar (PACC) file: rows after the dashed separator line, columns
    'alpha CL CD CDp CM ...'. Returns a dict alpha -> (CL, CM)."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("---")) + 1
    out = {}
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) >= 5:
            alpha, cl, cm = float(parts[0]), float(parts[1]), float(parts[4])
            out[round(alpha, 3)] = (cl, cm)
    return out
