"""Streamlit app smoke test: the script runs headless with default inputs and raises nothing."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_runs_with_defaults():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_app_flight_mode():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[1].set_value("Flight conditions").run()
    assert not at.exception
    assert any("M = " in md.value for md in at.sidebar.markdown)
