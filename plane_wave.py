"""
Compute strain rate field in 3D for plane-waves.
"""

import numpy as np
import xarray as xr


def to_spherical(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rho)
    phi = np.arctan2(y, x)
    return rho, theta, phi


def to_carthesian(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return x, y, z


def to_incidence(p, c):
    return xr.apply_ufunc(np.emath.arcsin, c * p)


def to_slowness(i, c):
    return np.sin(i) / c


def horizontal_slowness(px, py):
    return np.sqrt(px**2 + py**2)


def vertical_slowness(px, py, c, direction):
    p = horizontal_slowness(px, py)
    return direction * xr.apply_ufunc(np.emath.sqrt, 1 / c**2 - p**2)


def scattering_matrix(p, alpha, beta):
    """
    Compute the scattering matrix for a free surface boundary condition
    following Aki & Richards equations 5.27, 5.28, 5.31, 5.32 but with a
    different convention for SV-waves (opposite signe for upgoing SV-waves).
    """
    i = to_incidence(p, alpha)
    j = to_incidence(p, beta)
    S = {
        "P": {"P": 0, "SV": 0, "SH": 0},
        "SV": {"P": 0, "SV": 0, "SH": 0},
        "SH": {"P": 0, "SV": 0, "SH": 0},
    }
    a = 1.0 / beta**2 - 2.0 * p**2
    b = p * np.cos(i) / beta
    c = p * np.cos(j) / alpha
    d = (a**2 + 4.0 * b * c)
    S["P"]["P"] = (-a**2 + 4.0 * b * c) / d
    S["SV"]["P"] = - 4.0 * c * a / d  # opposite signe
    S["P"]["SV"] = 4.0 * b * a / d
    S["SV"]["SV"] = - (a**2 - 4.0 * b * c) / d  # opposite signe
    S["SH"]["SH"] = 1.0
    return S


def strain_rate_amplitude(px, py, pz, omega):
    return omega**2 * np.sqrt(np.real(px**2 + py**2 + pz**2))


def strain_rate_gain(ux, uy, uz, px, py, pz, kind):
    _, pt, pp = to_spherical(px, py, pz)
    _, ut, up = to_spherical(ux, uy, uz)
    dp = pp - up
    c_pt = np.cos(pt)
    s_pt = np.sin(pt)
    c_ut = np.cos(ut)
    s_ut = np.sin(ut)
    c_dp = np.cos(dp)
    s_dp = np.sin(dp)
    if kind == "P":
        return ((s_pt * s_ut * c_dp
                 + c_pt * c_ut)**2)
    elif kind == "SV":
        return (-s_pt**2 * s_ut * c_ut * c_dp
                + s_pt * s_ut**2 * c_pt * c_dp**2
                - s_pt * c_pt * c_ut**2 + s_ut * c_pt**2 * c_ut * c_dp)
    elif kind == "SH":
        return (-(s_pt * s_ut * c_dp
                  + c_pt * c_ut) * s_ut * s_dp)


def strain_rate_phase(x, y, z, px, py, pz, omega):
    return np.exp(1j * omega * (px * x + py * y + pz * z))


def strain_rate(x, y, z, ux, uy, uz, px, py, pz, omega, kind):
    return (strain_rate_amplitude(px, py, pz, omega)
            * strain_rate_gain(ux, uy, uz, px, py, pz, kind)
            * strain_rate_phase(x, y, z, px, py, pz, omega))


def infinite_space(x, y, z, ux, uy, uz, px, py, omega, c, kind, direction):
    pz = vertical_slowness(px, py, c, direction)
    out = strain_rate(x, y, z, ux, uy, uz, px, py, pz, omega, kind)
    return out


def half_space(x, y, z, ux, uy, uz, px, py, omega, alpha, beta, kind):
    c = {"P": alpha, "SV": beta, "SH": beta}
    ph = horizontal_slowness(px, py)
    S = scattering_matrix(ph, alpha, beta)
    out = infinite_space(x, y, z, ux, uy, uz, px, py, omega, c[kind], kind, -1)
    for converted in ["P", "SV", "SH"]:
        out += (S[kind][converted]
                * infinite_space(x, y, z, ux, uy, uz, px, py, omega, c[converted], converted, 1))
    return out
