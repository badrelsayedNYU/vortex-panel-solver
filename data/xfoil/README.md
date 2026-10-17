# XFOIL reference data (spec §5.3)

These files let the panel solver be checked against **XFOIL** (Drela, MIT), the standard
airfoil code. XFOIL is used **only** for this validation plot; the solver never runs it.

You will produce **8 small text files** (about 15 minutes):

| File | Contents |
|---|---|
| `naca0012_a0.cp`, `naca0012_a4.cp`, `naca0012_a8.cp` | surface Cp of NACA 0012 at α = 0°, 4°, 8° |
| `naca2412_a0.cp`, `naca2412_a4.cp`, `naca2412_a8.cp` | surface Cp of NACA 2412 at α = 0°, 4°, 8° |
| `naca0012_polar.txt`, `naca2412_polar.txt` | C_l and C_m at those three angles |

---

## Step 1: Get XFOIL (Windows)

XFOIL has no ready-made Mac version, so use a Windows PC (an NYU lab machine is fine).

1. Go to **https://web.mit.edu/drela/Public/web/xfoil/**
2. Under the downloads, click **XFOIL 6.99 for Windows** (a file named like `XFOIL6.99.zip`).
3. Unzip it to a short folder, e.g. **`C:\xfoil`**. Inside you should see `xfoil.exe`.

## Step 2: Copy the geometry files

1. From this folder, copy the **`geometry`** folder (it holds `naca0012_closedTE.dat` and
   `naca2412_closedTE.dat`) onto a USB stick or cloud drive.
2. Put it inside `C:\xfoil`, so the files are at `C:\xfoil\geometry\naca0012_closedTE.dat`.

## Step 3: Start XFOIL

1. Click **Start**, type **`cmd`**, press **Enter**. A black Command Prompt window opens.
2. Type these two lines (Enter after each):
   ```
   cd C:\xfoil
   xfoil
   ```
3. You will see an `XFOIL c>` prompt. XFOIL is now waiting for commands.

## Step 4: NACA 0012

Type each line and press **Enter**. The right column says what it does.

| Type this | What it does |
|---|---|
| `LOAD geometry/naca0012_closedTE.dat` | Loads the exact shape the panel solver uses |
| `PANE` | Re-spaces XFOIL's own panel nodes on that shape (the shape does not change) |
| `OPER` | Enters the analysis menu; the prompt becomes `.OPERi c>` (the `i` = inviscid) |
| `PACC` | Starts saving a polar (lift/moment table) |
| `naca0012_polar.txt` | Name of the polar file |
| *(just press Enter)* | No "dump" file needed |
| `ALFA 0` | Solves at α = 0°; prints CL and Cm |
| `CPWR naca0012_a0.cp` | Writes the surface Cp at this angle |
| `ALFA 4` | Solves at α = 4° |
| `CPWR naca0012_a4.cp` | Writes Cp |
| `ALFA 8` | Solves at α = 8° |
| `CPWR naca0012_a8.cp` | Writes Cp |
| `PACC` | Stops saving the polar (closes the file) |
| *(just press Enter)* | Leaves the OPER menu, back to `XFOIL c>` |

**Write down** the CL that XFOIL prints after each `ALFA` line. At α = 4° expect about 0.48.

## Step 5: NACA 2412

Repeat Step 4 exactly, with `2412` in place of `0012` in every file name
(the first line becomes `LOAD geometry/naca2412_closedTE.dat`). At α = 4° expect CL about 0.74.

Then type `QUIT` to close XFOIL.

## Step 6: Check the files

1. Open `C:\xfoil` in File Explorer. You should see the **8 files** listed at the top of this page.
2. Open `naca0012_a4.cp` in Notepad. It should look roughly like:
   ```
   #    x        y        Cp
     1.00000  0.00000   0.2...
     0.99...  0.00...   0.1...
   ```
3. Open `naca0012_polar.txt`. Near the bottom, under a line of dashes, there are rows like:
   ```
     alpha    CL        CD       CDp       CM   ...
     0.000   0.0000   0.00000   ...      0.0000
     4.000   0.48..   ...
   ```
   CD is 0 because the run is inviscid. That is expected.

## Step 7: Bring them to the Mac

Copy the 8 files into this folder on the Mac:
`~/Desktop/Projects/vortex-panel-solver/data/xfoil/`

Then run `python scripts/xfoil_compare.py` (or ask for it to be run). It writes
`figures/xfoil_comparison.png` and `results/xfoil_comparison.md`.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `File not found` after LOAD | Check the path: run `dir geometry` in Command Prompt; the files must be in `C:\xfoil\geometry` |
| XFOIL asks `Enter airfoil name` | Type `NACA 0012` (or `2412`) and press Enter |
| `File exists. Overwrite?` | Type `Y` |
| Prompt is `.OPERv c>` (with a `v`) | Viscous mode is on; type `VISC` once to switch back to inviscid (`.OPERi c>`) |
| A plot window pops up | Ignore it; it does not affect the files |
| CL far from 0.48 at α = 4° for NACA 0012 | Probably the wrong file was loaded; `QUIT` and start again |

---

## Understanding what this checks (for questions)

- **Why inviscid?** The panel method is inviscid, so XFOIL is run without its boundary
  layer. That compares like with like: any difference comes from the numerical method,
  not from viscous physics.
- **Why load our geometry instead of typing `NACA 0012` in XFOIL?** The solver uses the
  closed-trailing-edge NACA shape. Using the identical shape file means differences in
  trailing-edge closure or point placement cannot creep in and be mistaken for method error.
- **What does `PANE` do?** It keeps the shape but redistributes XFOIL's panel nodes using
  its own curvature-based spacing. So the two codes discretize the same shape differently,
  which is a fair test.
- **How does XFOIL's inviscid solver relate to this one?** Both are linear-vorticity panel
  methods with a Kutta condition. XFOIL uses a stream-function formulation and its own
  trailing-edge treatment (Drela, 1989); this solver uses the Kuethe & Chow velocity
  formulation. Same physics, different discretization.
- **What should the comparison show?** C_l agreeing to within a fraction of a percent
  and C_p curves lying on top of each other. Small differences are expected right at the
  leading-edge suction peak (where C_p converges slowly with panel count) and at the
  trailing edge.
- **What would a bad result mean?** A consistent C_l offset would suggest different
  geometry; a mismatch everywhere would suggest a sign or angle problem. The solver is
  already checked against an exact solution, so XFOIL is an independent confirmation
  on standard NACA sections.

---

## Optional: viscous polar for the drag plot (spec §6)

Same steps up to `OPER`, then:

| Type this | What it does |
|---|---|
| `VISC 1e6` | Turns on XFOIL's boundary layer at Re = 1,000,000 |
| `PACC` | Start saving a polar |
| `naca0012_visc_re1e6.txt` | File name |
| *(Enter)* | No dump file |
| `ASEQ -4 10 1` | Runs α = −4° to 10° in 1° steps |
| `PACC` | Stop saving |

Copy `naca0012_visc_re1e6.txt` here too; `scripts/drag_polar.py` overlays it automatically.
