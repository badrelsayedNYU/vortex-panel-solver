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

## On a Mac (XFOIL is already installed on this machine)

XFOIL 6.99 was compiled from the official MIT source into `~/Tools/Xfoil` and is available
as the `xfoil` command. The files are written straight into this folder, so nothing needs copying.

### Quick way (recommended): one command

`run_inviscid.xfoil` in this folder contains every command from Steps 2–4 below, in order.
In Terminal:
```
cd ~/Desktop/Projects/vortex-panel-solver/data/xfoil
xfoil < run_inviscid.xfoil > xfoil_run.log
ls
```
XFOIL reads the commands from the file, writes the 8 files, and quits (a few seconds).
Steps 1–5 below do the same thing by hand and explain each command.

### Step 1: Open Terminal in this folder

1. Press **⌘ + Space**, type **Terminal**, press **Enter**.
2. Type (Enter after each line):
   ```
   cd ~/Desktop/Projects/vortex-panel-solver/data/xfoil
   xfoil
   ```
3. You will see an `XFOIL c>` prompt.

### Step 2: Turn off plot windows (once per session)

XFOIL normally draws plots in an X11 window, which this Mac does not have. Type:

| Type this | What it does |
|---|---|
| `PLOP` | Opens the plot-options menu |
| `G` | Turns graphics off |
| *(just press Enter)* | Back to `XFOIL c>` |

### Step 3: NACA 0012

| Type this | What it does |
|---|---|
| `LOAD geometry/naca0012_closedTE.dat` | Loads the exact shape the panel solver uses |
| `PANE` | Re-spaces XFOIL's own 160 panel nodes on that shape (the shape does not change) |
| `OPER` | Enters the analysis menu; the prompt becomes `.OPERi c>` (the `i` = inviscid) |
| `PACC` | Starts saving a polar (lift/moment table) |
| `naca0012_polar.txt` | Name of the polar file |
| *(just press Enter)* | No "dump" file needed |
| `ALFA 0` | Solves at α = 0° |
| `CPWR naca0012_a0.cp` | Writes the surface Cp at this angle |
| `ALFA 4` | Solves at α = 4° |
| `CPWR naca0012_a4.cp` | Writes Cp |
| `ALFA 8` | Solves at α = 8° |
| `CPWR naca0012_a8.cp` | Writes Cp |
| `PACC` | Stops saving the polar (closes the file) |
| *(just press Enter)* | Leaves OPER, back to `XFOIL c>` |

After `LOAD` you will see **"Current airfoil cannot be set."** That is normal: the file has 401
points, more than XFOIL keeps directly, which is exactly why `PANE` comes next.

### Step 4: NACA 2412

Repeat Step 3 with `2412` in place of `0012` in every file name
(first line: `LOAD geometry/naca2412_closedTE.dat`). Then type `QUIT`.

### Step 5: Check the results

Still in Terminal, type:
```
ls
cat naca0012_polar.txt
```
`ls` should list the 8 files. At the bottom of the polar file, under a line of dashes:
```
   alpha    CL        CD       CDp       CM   ...
   0.000   0.0000   0.00000  ...      0.0000
   4.000   0.48..   0.00000  ...     -0.005.
   8.000   0.96..   ...
```
CD is 0 because the run is inviscid. For NACA 2412 expect CL about 0.26, 0.74 and 1.22.

### Step 6: Run the comparison

Back in the project folder: `python scripts/xfoil_compare.py` (or ask for it to be run).
It writes `figures/xfoil_comparison.png` and `results/xfoil_comparison.md`.

---

## On Windows (alternative)

1. Download **XFOIL 6.99 for Windows** from https://web.mit.edu/drela/Public/web/xfoil/ and unzip it to `C:\xfoil`.
2. Copy this folder's `geometry` folder into `C:\xfoil`.
3. Start → `cmd` → Enter, then `cd C:\xfoil` and `xfoil`.
4. Follow Steps 3–5 above (Windows XFOIL opens plot windows; Step 2 is optional there).
5. Copy the 8 files into `data/xfoil/` on the Mac.

## If something goes wrong

| What you see | What to do |
|---|---|
| `File not found` after LOAD | You are not in the right folder: `QUIT`, then `cd ~/Desktop/Projects/vortex-panel-solver/data/xfoil` and start again (Windows: files must be in `C:\xfoil\geometry`) |
| XFOIL asks `Enter airfoil name` | Type `NACA 0012` (or `2412`) and press Enter |
| `File exists. Overwrite?` | Type `Y` |
| Prompt is `.OPERv c>` (with a `v`) | Viscous mode is on; type `VISC` once to switch back to inviscid (`.OPERi c>`) |
| `Cannot open display` or similar X11 message (Mac) | Graphics are still on: do Step 2 (`PLOP`, `G`, Enter) |
| CL does not appear on screen after `ALFA` | Normal with graphics off; the values are in the polar file (Step 5) |
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
