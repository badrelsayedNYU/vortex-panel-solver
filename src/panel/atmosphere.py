"""1976 US Standard Atmosphere, 0-20 km, and flight-condition conversion (spec §6b).

Altitude input is geometric; the model layers are defined in geopotential altitude
H = r0 h / (r0 + h), r0 = 6 356 766 m.
  troposphere       0 <= H < 11 km:  T = T0 - L H,   p = p0 (T/T0)^(g0/(R L))
  lower stratosphere 11 <= H <= 20 km: T = 216.65 K,  p = p11 exp(-g0 (H - 11000)/(R T))
  rho = p/(R T),  a = sqrt(gamma R T),  mu = 1.458e-6 T^1.5 / (T + 110.4)  (Sutherland)
(U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF.)
"""

import numpy as np

R = 287.05287        # J/(kg K), R*/M0 of the 1976 standard
G0 = 9.80665         # m/s^2
GAMMA = 1.4
T0, P0, LAPSE = 288.15, 101325.0, 0.0065
R_EARTH = 6356766.0
H_TROP = 11000.0
T_STRAT = T0 - LAPSE * H_TROP                       # 216.65 K
P_TROP = P0 * (T_STRAT / T0) ** (G0 / (R * LAPSE))  # 22632 Pa


def atmosphere(h):
    """Return dict of T [K], p [Pa], rho [kg/m^3], a [m/s], mu [Pa s] at geometric altitude h [m]."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(h > 20000):
        raise ValueError("altitude must be within 0-20 km (troposphere and lower stratosphere)")
    H = R_EARTH * h / (R_EARTH + h)
    T = np.where(H < H_TROP, T0 - LAPSE * H, T_STRAT)
    p = np.where(H < H_TROP, P0 * (T / T0) ** (G0 / (R * LAPSE)),
                 P_TROP * np.exp(-G0 * (H - H_TROP) / (R * T_STRAT)))
    rho = p / (R * T)
    return {"T": T, "p": p, "rho": rho, "a": np.sqrt(GAMMA * R * T),
            "mu": 1.458e-6 * T**1.5 / (T + 110.4)}


def flight_to_mach_re(V, h, c):
    """Flight speed V [m/s], altitude h [m], chord c [m] -> (Mach, Reynolds, atmosphere dict):
    M = V/a,  Re = rho V c / mu  (spec §6b)."""
    atm = atmosphere(h)
    return float(V / atm["a"]), float(atm["rho"] * V * c / atm["mu"]), atm
