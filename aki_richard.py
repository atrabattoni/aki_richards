import numpy as np


def zoeppritz(r1, a1, b1, r2, a2, b2, p, energy_normalization=False):
    # Angles computation
    i1 = np.arcsin(complex(p * a1))
    j1 = np.arcsin(complex(p * b1))
    i2 = np.arcsin(complex(p * a2))
    j2 = np.arcsin(complex(p * b2))

    I = [i1, j1, i2, j2]

    # Scattering matrix computation
    M = np.array([[-a1 * p, -np.cos(j1), a2 * p, np.cos(j2)],
                  [np.cos(i1), -b1 * p, np.cos(i2), -b2 * p],
                  [2 * r1 * b1**2 * p * np.cos(i1), r1 * b1 * (1 - 2 * b1**2 * p**2),
                   2 * r2 * b2**2 * p * np.cos(i2), r2 * b2 * (1 - 2 * b2**2 * p**2)],
                  [-r1 * a1 * (1 - 2 * b1**2 * p**2), 2 * r1 * b1**2 * p * np.cos(j1),
                   r2 * a2 * (1 - 2 * b2**2 * p**2), -2 * r2 * b2**2 * p * np.cos(j2)]])

    N = np.array([[a1 * p, np.cos(j1), -a2 * p, -np.cos(j2)],
                  [np.cos(i1), -b1 * p, np.cos(i2), -b2 * p],
                  [2 * r1 * b1**2 * p * np.cos(i1), r1 * b1 * (1 - 2 * b1**2 * p**2),
                   2 * r2 * b2**2 * p * np.cos(i2), r2 * b2 * (1 - 2 * b2**2 * p**2)],
                  [r1 * a1 * (1 - 2 * b1**2 * p**2), -2 * r1 * b1**2 * p * np.cos(j1),
                   -r2 * a2 * (1 - 2 * b2**2 * p**2), 2 * r2 * b2**2 * p * np.cos(j2)]])

    S = np.linalg.inv(M) @ N

    # Energy normalization if required
    if not energy_normalization:
        return I, S
    else:
        L = np.array([r1 * a1 * np.cos(i1),
                      r1 * b1 * np.cos(j1),
                      r2 * a2 * np.cos(i2),
                      r2 * b2 * np.cos(j2)])[..., np.newaxis]
        D = np.sqrt(L / L.T)
        return I, S * D
