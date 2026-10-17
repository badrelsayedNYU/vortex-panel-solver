"""1976 US Standard Atmosphere and flight-condition conversion (spec §6b).

OWNER TO VERIFY: the spec says these reference values come from a printed standard table. The
values below were entered from the 1976 table (geometric altitude) and agree
with the model to <0.05%; please check them against your table and correct any that differ.
"""

import pytest

from panel.atmosphere import atmosphere, flight_to_mach_re

TABLE = {  # h [m]: (T [K], p [Pa], rho [kg/m^3], a [m/s], mu [Pa s])
    0: (288.15, 101325.0, 1.2250, 340.29, 1.7894e-5),
    5000: (255.68, 54048.0, 0.73643, 320.55, 1.6282e-5),
    11000: (216.77, 22700.0, 0.36480, 295.15, 1.4223e-5),
}


@pytest.mark.parametrize("h", TABLE)
def test_matches_standard_table(h):
    T, p, rho, a, mu = TABLE[h]
    atm = atmosphere(h)
    assert atm["T"] == pytest.approx(T, rel=1e-4)
    assert atm["p"] == pytest.approx(p, rel=5e-4)
    assert atm["rho"] == pytest.approx(rho, rel=5e-4)
    assert atm["a"] == pytest.approx(a, rel=2e-4)
    assert atm["mu"] == pytest.approx(mu, rel=5e-4)


def test_flight_conversion():
    M, Re, atm = flight_to_mach_re(V=50.0, h=1000.0, c=0.3)
    assert M == pytest.approx(50.0 / atm["a"])
    assert Re == pytest.approx(atm["rho"] * 50.0 * 0.3 / atm["mu"])
    assert 0.14 < M < 0.15 and 9e5 < Re < 1.05e6


def test_out_of_range_altitude_rejected():
    with pytest.raises(ValueError):
        atmosphere(25000.0)
