"""Streamlit app smoke test: the script runs headless with default inputs and raises nothing."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_runs_with_defaults():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 9


def test_app_flight_mode():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[1].set_value("Flight conditions").run()
    assert not at.exception
    assert any("M = " in md.value for md in at.sidebar.markdown)


def test_app_exposes_new_studies_and_runs_flow_field():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert {tab.label for tab in at.tabs} >= {"Flow field", "Validation", "Convergence"}
    next(b for b in at.button if b.label == "Compute flow field").click().run()
    assert not at.exception


def test_app_xfoil_reference_switch():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    next(r for r in at.radio if r.label == "Validation reference").set_value("XFOIL inviscid").run()
    assert not at.exception
    assert any(s.label == "Reference airfoil" for s in at.selectbox)
