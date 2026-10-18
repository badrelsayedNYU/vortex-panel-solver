# XFOIL reference data

These eight files contain XFOIL 6.99 inviscid results for NACA 0012 and NACA 2412,
using the closed-trailing-edge coordinates in [`geometry/`](geometry/).

- `naca0012_a0.cp`, `naca0012_a4.cp`, `naca0012_a8.cp`: surface pressure
  coefficients at 0°, 4°, and 8° angle of attack for NACA 0012.
- `naca2412_a0.cp`, `naca2412_a4.cp`, `naca2412_a8.cp`: the corresponding
  pressure coefficients for NACA 2412.
- `naca0012_polar.txt`, `naca2412_polar.txt`: the integrated coefficients
  at those angles for each airfoil.

[`run_inviscid.xfoil`](run_inviscid.xfoil) is the XFOIL command file that produced
these results. It loads each geometry, repanels it, and writes the pressure and
polar files. To reproduce the runs with XFOIL installed, run from this directory:

```sh
xfoil < run_inviscid.xfoil
```

The panel-solver comparison is shown in
[`figures/xfoil_comparison.png`](../../figures/xfoil_comparison.png), with the
coefficient comparison in
[`results/xfoil_comparison.md`](../../results/xfoil_comparison.md).
