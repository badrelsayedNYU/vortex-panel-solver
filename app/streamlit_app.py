"""Web interface for the vortex panel solver (spec §6b). UI only: all physics lives in
panel.analysis / panel.plots / panel.atmosphere."""

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from panel import plots  # noqa: E402
from panel.analysis import cp_distributions, make_airfoil, summary, sweep  # noqa: E402
from panel.atmosphere import flight_to_mach_re  # noqa: E402

st.set_page_config(page_title="Vortex Panel Airfoil Analysis", layout="wide")
st.title("Vortex panel airfoil analysis")
st.caption("Linear-strength vortex panel method with Kármán–Tsien compressibility correction and an "
           "integral boundary layer (Thwaites → Michel → Head → Squire–Young). Everything is computed "
           "by this project's own solver.")
st.info("**Physics note.** In the inviscid, incompressible solution, C_l and Cp do not depend on "
        "velocity. Velocity enters only through Mach (compressibility correction) and Reynolds "
        "(boundary layer). The inviscid method does **not** predict stall: C_l keeps rising with α.")


@st.cache_data(show_spinner="Solving...")
def run(naca, dat_bytes, n, alphas, machs, reynolds, x_trip, cp_alphas):
    dat_path = None
    if dat_bytes is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp.write(dat_bytes)
        tmp.close()
        dat_path = tmp.name
    airfoil = make_airfoil(naca if dat_path is None else None, dat_path, n)
    df = sweep(airfoil, alphas, machs, reynolds, x_trip)
    cpdf = cp_distributions(airfoil, cp_alphas, machs)
    return airfoil, df, cpdf


with st.sidebar:
    st.header("Airfoil")
    source = st.radio("Geometry", ["NACA 4-digit", "Upload Selig .dat"], horizontal=True)
    naca, dat_bytes = None, None
    if source == "NACA 4-digit":
        naca = st.text_input("NACA code", "2412", max_chars=4)
    else:
        upload = st.file_uploader("Selig-format coordinates", type=["dat", "txt"])
        dat_bytes = upload.getvalue() if upload else None
    n = st.select_slider("Panels N", options=[80, 120, 160, 200, 240, 320], value=160)

    st.header("Angle of attack (deg)")
    c1, c2, c3 = st.columns(3)
    a0 = c1.number_input("start", value=-4.0, step=1.0)
    a1 = c2.number_input("stop", value=12.0, step=1.0)
    da = c3.number_input("step", value=1.0, min_value=0.1, step=0.5)

    st.header("Flow conditions")
    mode = st.radio("Specify", ["Mach & Reynolds", "Flight conditions"], horizontal=True)
    viscous = st.toggle("Boundary layer (drag)", value=True)
    flight = None
    if mode == "Mach & Reynolds":
        machs = [float(v) for v in st.text_input("Mach numbers", "0, 0.3, 0.5").split(",") if v.strip()]
        reynolds = [float(v) for v in st.text_input("Reynolds numbers", "1e6").split(",") if v.strip()]
    else:
        V = st.number_input("Speed V (m/s)", value=50.0, min_value=1.0)
        h = st.number_input("Altitude h (m)", value=1000.0, min_value=0.0, max_value=20000.0)
        c = st.number_input("Chord c (m)", value=0.3, min_value=0.01)
        M, Re, atm = flight_to_mach_re(V, h, c)
        flight = dict(V=V, c=c, rho=float(atm["rho"]))
        st.write(f"→ M = **{M:.3f}**, Re = **{Re:.3g}**, ρ = {float(atm['rho']):.4f} kg/m³")
        machs, reynolds = [M], [Re]
    trip = st.number_input("Force transition at x/c (0 = free transition)", value=0.0,
                           min_value=0.0, max_value=0.99, step=0.05) if viscous else 0.0
    cp_text = st.text_input("α values for Cp plots", "0, 4, 8")

alphas = [round(a0 + k * da, 6) for k in range(int(round((a1 - a0) / da)) + 1)]
cp_alphas = [float(v) for v in cp_text.split(",") if v.strip()]
if source == "Upload Selig .dat" and dat_bytes is None:
    st.warning("Upload a Selig-format .dat file to start.")
    st.stop()

try:
    airfoil, df, cpdf = run(naca, dat_bytes, n, tuple(alphas), tuple(machs),
                            tuple(reynolds) if viscous else None, trip or None, tuple(cp_alphas))
except ValueError as err:
    st.error(str(err))
    st.stop()

if flight:
    q = 0.5 * flight["rho"] * flight["V"] ** 2
    df["lift_per_span_N_m"] = q * flight["c"] * df["cl"]
    if "cd" in df:
        df["drag_per_span_N_m"] = q * flight["c"] * df["cd"]

n_invalid = int((~df["compressibility_valid"]).sum())
if n_invalid:
    st.warning(f"{n_invalid} points are at or above the critical Mach number: the compressibility "
               "correction is invalid there (shown as grey ×).")
if "separated" in df and df["separated"].any():
    st.warning(f"{int(df['separated'].sum())} points separate before the trailing edge: drag and L/D "
               "are unreliable there (open red circles).")

tabs = st.tabs(["Geometry", "Lift & moment", "Pressure", "Critical Mach", "Drag", "Summary & data"])
with tabs[0]:
    st.pyplot(plots.plot_geometry(airfoil))
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.pyplot(plots.plot_cl_alpha(df))
    c2.pyplot(plots.plot_cm_alpha(df))
with tabs[2]:
    st.pyplot(plots.plot_cp(cpdf))
with tabs[3]:
    st.pyplot(plots.plot_mcr_alpha(df))
with tabs[4]:
    if "cd" in df:
        c1, c2 = st.columns(2)
        c1.pyplot(plots.plot_drag_polar(df))
        c2.pyplot(plots.plot_cd_alpha(df))
        c1.pyplot(plots.plot_ld_alpha(df))
        c2.pyplot(plots.plot_transition(df))
    else:
        st.write("Turn on the boundary layer in the sidebar to compute drag.")
with tabs[5]:
    summ = summary(df)
    st.table(pd.Series({k: f"{v:.4g}" for k, v in summ.items()}, name="value"))
    st.dataframe(df, width="stretch")
    st.download_button("Download sweep (CSV)", df.to_csv(index=False), "sweep.csv", "text/csv")
    st.download_button("Download summary (JSON)", json.dumps(summ, indent=2, default=float),
                       "summary.json", "application/json")
