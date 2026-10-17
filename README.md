# Vortex Panel Method Airfoil Solver

A linear-strength vortex panel solver for 2D airfoils, with compressibility corrections, a critical-Mach analysis, and an airfoil analysis tool built on top.

> **Work in progress.** Done: geometry (NACA 4-digit, Selig `.dat`, Kármán–Trefftz) and the core panel solve (Cp, C_l, C_m). In progress: validation, compressibility, streamlines, and the analysis tool.

## Features
- NACA 4-digit and Selig `.dat` geometry with cosine panel spacing
- Linear-strength vortex panels (Kuethe & Chow formulation), flow tangency + Kutta condition
- Cp distributions, C_l–α, C_m, and streamlines
- Prandtl–Glauert and Kármán–Tsien corrections; critical Mach number
- Validation cases: exact Kármán–Trefftz solution, thin airfoil theory, and XFOIL
- Airfoil analysis command-line tool and web app built on the solver

![Airfoil geometry](figures/geometry.png)

## Run it
```bash
pip install -e ".[dev]"
pytest
python scripts/make_figures.py
```

## References
- Kuethe & Chow, *Foundations of Aerodynamics*, 5th ed., §5.10
- Katz & Plotkin, *Low-Speed Aerodynamics*
- Anderson, *Fundamentals of Aerodynamics*
