"""Airfoil geometry: NACA 4-digit generator, Selig .dat loader with repaneling, and the
Kármán–Trefftz conformal-map airfoil used for exact validation (spec §2).

Convention (spec §2.1): N panels, N+1 nodes, ordered clockwise from the trailing edge on
the lower surface -> leading edge -> upper surface -> trailing edge.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


@dataclass
class Airfoil:
    name: str
    X: np.ndarray          # node x, shape (N+1,)
    Y: np.ndarray          # node y, shape (N+1,)
    meta: dict = field(default_factory=dict)

    @property
    def n_panels(self):
        return self.X.size - 1

    def panels(self):
        """Panel lengths S_j, orientations theta_j and midpoint control points (spec §3.1).

        S_j = |(X_{j+1}, Y_{j+1}) - (X_j, Y_j)|,  theta_j = atan2(Y_{j+1} - Y_j, X_{j+1} - X_j),
        (x_j, y_j) = midpoint of panel j.
        """
        dx, dy = np.diff(self.X), np.diff(self.Y)
        S = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        xc = 0.5 * (self.X[:-1] + self.X[1:])
        yc = 0.5 * (self.Y[:-1] + self.Y[1:])
        return S, theta, xc, yc


# ---------------------------------------------------------------- NACA 4-digit

def cosine_spacing(n_points):
    """x = (1 - cos beta)/2 with beta uniform on [0, pi]  (spec §2.1); clusters at LE and TE."""
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def naca4_thickness(x, t, closed_te=False):
    """Half-thickness y_t = 5t(0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 - a4 x^4)
    with a4 = 0.1015 (standard, open TE) or 0.1036 (closed TE)  (spec §2.1; Abbott & von Doenhoff §6.4)."""
    a4 = 0.1036 if closed_te else 0.1015
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - a4 * x**4)


def naca4_camber(x, m, p):
    """Mean line y_c and slope dy_c/dx  (spec §2.1; Abbott & von Doenhoff §6.4).

    y_c = m/p^2 (2px - x^2)                    for x < p
    y_c = m/(1-p)^2 ((1 - 2p) + 2px - x^2)     for x >= p
    dy_c/dx = 2m/p^2 (p - x)  or  2m/(1-p)^2 (p - x)
    """
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    fore = x < p
    yc = np.where(fore, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(fore, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def naca4(code, n_panels=160, closed_te=False):
    """NACA 'mpxx' airfoil with N = n_panels panels (N even), cosine-spaced, clockwise from the
    lower-surface TE  (spec §2.1).

    Surfaces: x_U = x - y_t sin(theta_c), y_U = y_c + y_t cos(theta_c);
              x_L = x + y_t sin(theta_c), y_L = y_c - y_t cos(theta_c);  theta_c = atan(dy_c/dx).
    """
    code = str(code).zfill(4)
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"expected a 4-digit NACA code, got {code!r}")
    if n_panels % 2:
        raise ValueError("n_panels must be even (equal panels on each surface)")
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100

    x = cosine_spacing(n_panels // 2 + 1)
    yt = naca4_thickness(x, t, closed_te)
    yc, dyc = naca4_camber(x, m, p)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)

    X = np.concatenate([xl[::-1], xu[1:]])  # lower TE -> LE, then LE -> upper TE
    Y = np.concatenate([yl[::-1], yu[1:]])
    return Airfoil(f"NACA {code}", X, Y, {"m": m, "p": p, "t": t, "closed_te": closed_te})


# ---------------------------------------------------------------- Selig .dat files

def read_selig(path):
    """Read a Selig-format coordinate file: a name line, then x y pairs running from the upper
    TE over the LE to the lower TE (counter-clockwise). Returns (name, x, y) in file order."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    name = lines[0]
    pts = np.array([[float(v) for v in ln.split()[:2]] for ln in lines[1:]])
    if pts[0, 0] > 1.5:  # Lednicer files start with point counts, e.g. "61. 61."
        raise ValueError(f"{path}: looks like Lednicer format; only Selig format is supported")
    return name, pts[:, 0], pts[:, 1]


def repanel(x, y, n_panels=160, name="airfoil"):
    """Repanel ordered surface points with cosine spacing in arc length  (spec §2.2).

    The input (any closed-ish ordering running TE -> LE -> TE) is fitted with cubic splines
    x(s), y(s) in cumulative chord length s. The LE is the point of minimum x(s). Each surface
    gets N/2 panels with s = s_0 + (s_1 - s_0)(1 - cos beta)/2, clustering nodes at LE and TE.
    The result is reordered clockwise from the lower-surface TE if needed.
    """
    if n_panels % 2:
        raise ValueError("n_panels must be even")
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sx, sy = CubicSpline(s, x), CubicSpline(s, y)
    i_min = int(np.argmin(x))
    lo, hi = s[max(i_min - 2, 0)], s[min(i_min + 2, s.size - 1)]
    s_le = minimize_scalar(sx, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-12}).x

    half = cosine_spacing(n_panels // 2 + 1)
    s_new = np.concatenate([half * s_le, s_le + half[1:] * (s[-1] - s_le)])
    X, Y = sx(s_new), sy(s_new)
    if signed_area(X, Y) > 0:  # counter-clockwise (e.g. Selig order): reverse
        X, Y = X[::-1], Y[::-1]
    return Airfoil(name, X, Y)


def load_selig(path, n_panels=160):
    """Load a Selig .dat file and repanel it with cosine spacing (spec §2.2)."""
    name, x, y = read_selig(path)
    return repanel(x, y, n_panels, name)


def write_selig(airfoil, path):
    """Write nodes in Selig order (upper TE -> LE -> lower TE)."""
    lines = [airfoil.name] + [f"{xi: .8f} {yi: .8f}" for xi, yi in zip(airfoil.X[::-1], airfoil.Y[::-1])]
    Path(path).write_text("\n".join(lines) + "\n")


def signed_area(X, Y):
    """Shoelace area of the closed polygon through the nodes; negative for clockwise ordering."""
    return 0.5 * np.sum(X * np.roll(Y, -1) - np.roll(X, -1) * Y)


# ---------------------------------------------------------------- Kármán–Trefftz

@dataclass
class KarmanTrefftz:
    """Kármán–Trefftz map z = n b (1 + w)/(1 - w),  w = ((zeta - b)/(zeta + b))^n,  n = 2 - tau/pi
    (Katz & Plotkin §6.7; Milne-Thomson). The circle |zeta - mu| = R passes through zeta = b,
    which maps to the sharp trailing edge with included angle tau. n = 2 (tau = 0) is Joukowski.

    The ratio form keeps the complex power's branch cut away from the circle: the image of the
    circle under (zeta - b)/(zeta + b) is a circle through 0 that meets the real axis only at
    0 and at a positive point.
    """
    mu: complex = complex(-0.1, 0.1)   # circle center (b = 1)
    tau_deg: float = 10.0              # trailing-edge angle
    b: float = 1.0

    @property
    def n(self):
        return 2.0 - np.deg2rad(self.tau_deg) / np.pi

    @property
    def R(self):
        return abs(self.b - self.mu)

    @property
    def beta(self):
        """Angle with b - mu = R exp(-i beta); beta > 0 for a cambered (mu_y > 0) section."""
        return -np.angle(self.b - self.mu)

    def z(self, zeta):
        w = ((zeta - self.b) / (zeta + self.b)) ** self.n
        return self.n * self.b * (1 + w) / (1 - w)

    def dz_dzeta(self, zeta):
        """dz/dzeta = 4 n^2 b^2 w / ((1 - w)^2 (zeta^2 - b^2)); zero at the TE (zeta = b)."""
        w = ((zeta - self.b) / (zeta + self.b)) ** self.n
        return 4 * self.n**2 * self.b**2 * w / ((1 - w) ** 2 * (zeta**2 - self.b**2))

    def circle(self, phi):
        return self.mu + self.R * np.exp(1j * phi)

    def chord_transform(self, n_fine=20001):
        """(x_le, chord) of the mapped airfoil, used to normalise it to x/c in [0, 1]."""
        phi = np.linspace(0, 2 * np.pi, n_fine)
        zz = self.z(self.circle(phi))
        x_le, x_te = zz.real.min(), (self.n * self.b)
        return x_le, x_te - x_le

    def airfoil(self, n_panels=160):
        """Nodes from circle angles phi_k = phi_TE - 2 pi k/N (clockwise; uniform in phi gives
        natural LE/TE clustering after mapping), normalised so x/c in [0, 1] with TE at (1, 0).
        The node angles are stored in meta['phi'] for comparison with the exact solution."""
        phi_te = -self.beta
        phi = phi_te - 2 * np.pi * np.arange(n_panels + 1) / n_panels
        zz = self.z(self.circle(phi))
        zz[0] = zz[-1] = self.n * self.b  # exact TE (avoids 0/0 round-off in w at zeta = b)
        x_le, c = self.chord_transform()
        X, Y = (zz.real - x_le) / c, zz.imag / c
        return Airfoil(f"Kármán–Trefftz (mu = {self.mu}, tau = {self.tau_deg}°)", X, Y,
                       {"phi": phi, "kt": self, "x_le": x_le, "chord": c})
