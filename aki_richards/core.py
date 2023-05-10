import numpy as np


def scattering_matrix(r1, a1, b1, r2, a2, b2, p, energy_normalization=False):
    """
    Compute the scattering matrix at a solid/solid interface.

    See 5.2.4 `Reflection and Transmission of P-SV across a Solid-Solid Interface`
    in Aki & Richards.

    Parameters
    ----------
    r1 : float
        Top layer density.
    a1 : float
        Top layer P-wave seed (km/s).
    b1 : float
        Top layer S-wave speed (km/s).
    r2 : float
        Bottom layer density.
    a2 : float
        Bottom layer P-wave speed.
    b2 : float
        Bottom layer S-wave speed.
    p : float
        Slowness (s/km).
    energy_normalization : bool, optional
        Wether to apply energy normalization, by default False.

    Returns
    -------
    4x4 array
        Scattering matrix.
    """

    # Angles computation
    i1 = np.arcsin(complex(p * a1))
    j1 = np.arcsin(complex(p * b1))
    i2 = np.arcsin(complex(p * a2))
    j2 = np.arcsin(complex(p * b2))

    I = [i1, j1, i2, j2]

    # Scattering matrix computation
    M = np.array(
        [
            [-a1 * p, -np.cos(j1), a2 * p, np.cos(j2)],
            [np.cos(i1), -b1 * p, np.cos(i2), -b2 * p],
            [
                2 * r1 * b1**2 * p * np.cos(i1),
                r1 * b1 * (1 - 2 * b1**2 * p**2),
                2 * r2 * b2**2 * p * np.cos(i2),
                r2 * b2 * (1 - 2 * b2**2 * p**2),
            ],
            [
                -r1 * a1 * (1 - 2 * b1**2 * p**2),
                2 * r1 * b1**2 * p * np.cos(j1),
                r2 * a2 * (1 - 2 * b2**2 * p**2),
                -2 * r2 * b2**2 * p * np.cos(j2),
            ],
        ]
    )

    N = np.array(
        [
            [a1 * p, np.cos(j1), -a2 * p, -np.cos(j2)],
            [np.cos(i1), -b1 * p, np.cos(i2), -b2 * p],
            [
                2 * r1 * b1**2 * p * np.cos(i1),
                r1 * b1 * (1 - 2 * b1**2 * p**2),
                2 * r2 * b2**2 * p * np.cos(i2),
                r2 * b2 * (1 - 2 * b2**2 * p**2),
            ],
            [
                r1 * a1 * (1 - 2 * b1**2 * p**2),
                -2 * r1 * b1**2 * p * np.cos(j1),
                -r2 * a2 * (1 - 2 * b2**2 * p**2),
                2 * r2 * b2**2 * p * np.cos(j2),
            ],
        ]
    )

    S = np.linalg.inv(M) @ N

    # Energy normalization if required
    if not energy_normalization:
        return I, S
    else:
        L = np.array(
            [
                r1 * a1 * np.cos(i1),
                r1 * b1 * np.cos(j1),
                r2 * a2 * np.cos(i2),
                r2 * b2 * np.cos(j2),
            ]
        )[..., np.newaxis]
        D = np.sqrt(L / L.T)
        return I, S * D
