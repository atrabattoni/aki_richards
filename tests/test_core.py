import numpy as np
from aki_richards import scattering_matrix, interface_motion
import matplotlib.pyplot as plt

# Aki & Richard page 147
_, S = scattering_matrix(3, 6, 3.5, 4, 7, 4.2, 0.1, energy_normalization=True)
S_validate = np.array(
    [
        [0.1065, -0.1766, 0.9701, -0.1277],
        [-0.1766, -0.0807, 0.1326, 0.9720],
        [0.9701, 0.1326, -0.0567, 0.1950],
        [-0.1277, 0.9720, 0.1950, 0.0309],
    ]
)
assert np.allclose(S, S_validate, atol=1e-4)

# Other good sense validations
assert np.allclose(S @ np.conj(S.T), np.eye(4))
assert np.allclose(np.sum(S * np.conj(S), axis=-1), np.ones(4))

# Aki & Richard page 137
r1, a1, b1 = 0, 0, 0
r2, a2, b3 = 1, 5, 3
p = np.linspace(0, 1 / a2, 1000)

I = np.zeros((len(p), 4), dtype="complex")
S = np.zeros((len(p), 4, 4), dtype="complex")
for k, pk in enumerate(p):
    I[k], S[k] = scattering_matrix(0, 0, 0, 1, 5, 3, pk)

assert np.allclose(S.imag, 0)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(p, np.real(S[:, 2, 2]), label="PP")
ax.plot(p, np.real(S[:, 2, 3]), label="SP")
ax.plot(p, np.real(S[:, 3, 2]), label="PS")
ax.plot(p, np.real(S[:, 3, 3]), label="SS")
ax.set_ylabel("amplitude")
ax.set_xlabel("slowness")
ax.legend()
fig.show()

# Aki & Richard page 154
r1, a1, b1 = 0, 0, 0
r2, a2, b2 = 1, 5, 3
p = np.linspace(0, 1 / b2, 1000)

I = np.zeros((len(p), 4), dtype="complex")
S = np.zeros((len(p), 4, 4), dtype="complex")
for k, pk in enumerate(p):
    I[k], S[k] = scattering_matrix(0, 0, 0, 1, 5, 3, pk)

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(p, np.abs(S[:, 2, 3]), label="SP")
axes[0, 1].plot(p, np.abs(S[:, 3, 3]), label="SS")
axes[1, 0].plot(p, np.angle(S[:, 2, 3]) % (2 * np.pi), label="SP")
axes[1, 1].plot(p, np.angle(S[:, 3, 3]), label="SS")

for ax in axes.flat:
    ax.set_xlabel("slowness")
    ax.legend()

fig.show()

assert np.allclose(interface_motion(I, S, "direct"), interface_motion(I, S, "indirect"))