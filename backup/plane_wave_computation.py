# %%
import matplotlib.pyplot as plt
import numpy as np
from plane_wave import *

# %%
# Validate free_surface_scattering_matrix with the figure 5.6 of
# Aki & Richards.

alpha = 5
beta = 3
p = np.linspace(0, 1/alpha, 1001)
S = scattering_matrix(p, alpha, beta)

fig = plt.figure(dpi=144)

# Wide
ax = fig.add_subplot(121)
ax.axhline(0, color="grey", lw=0.5)
ax.plot(p, S["P"]["P"], "k", label="PP")
ax.plot(p, S["P"]["SV"], "k:", label="PS")
ax.plot(p, -S["SV"]["P"], "grey", label="SP")  # different convention
ax.plot(p, -S["SV"]["SV"], "k--", label="SS")  # different convention
ax.annotate(r"$\alpha$ = 5 km/s", (0.025, 4.5))
ax.annotate(r"$\beta$ = 3 km/s", (0.025, 4.5 - 0.4))
ax.set_xlim(0, 0.25)
ax.set_ylim(-1, 5)
ax.set_xlabel("Slowness p (s/km)")
ax.set_ylabel("Reflection/conversion coefficients")

# zoom
ax = fig.add_subplot(122)
ax.axhline(0, color="grey", lw=0.5)
ax.plot(p, S["P"]["P"], "k", label="PP")
ax.plot(p, S["P"]["SV"], "k:", label="PS")
ax.plot(p, -S["SV"]["P"], "grey", label="SP")  # different convention
ax.plot(p, -S["SV"]["SV"], "k--", label="SS")  # different convention
ax.legend()
ax.set_xlim(0.190, 0.205)
ax.set_ylim(-1, 5)
ax.set_xlabel("Slowness p (s/km)")

# %%
# Validate free_surface_scattering_matrix with the figure 5.10 of
# Aki & Richards.

alpha = 5
beta = 3
p = np.linspace(0, 1/beta, 1001)
S = scattering_matrix(p, alpha, beta)
sp = -S["SV"]["P"]  # different convention
ss = -S["SV"]["SV"]  # different convention

fig, axes = plt.subplots(2, 2, figsize=(5, 6), dpi=144)
ax = axes
ax[0, 0].plot(p, np.abs(sp), "k")
ax[0, 0].set_ylabel("|SP| Amplitude")

ax[1, 0].plot(p, np.unwrap(np.angle(sp)), "k")
ax[1, 0].set_ylabel("|SP| Phase advance")

ax[0, 1].plot(p, np.unwrap(np.abs(ss)), "k")
ax[0, 1].set_ylabel("|SS| Amplitude")

phase = np.angle(ss)
phase[phase == np.pi] = - np.pi
ax[1, 1].plot(p, phase, "k")
ax[1, 1].set_ylabel("|SS| Phase advance")
for ax in axes.flat:
    ax.set_xlim(0, 1/beta)
for ax in axes[1, :]:
    ax.set_xlabel("Slowness, p")
fig.tight_layout()

# %%
# Vertical directivity

alpha = 5
beta = 3

fig = plt.figure(figsize=(10, 4), dpi=144)

# P-waves
p = np.linspace(-1/alpha, 1/alpha, 1001)
i = to_incidence(p, alpha)
infspc = infinite_space(0, 0, 0, 1, 0, 0, p, 0, 1, alpha, "P", -1)
hlfspc = half_space(0, 0, 0, 1, 0, 0, p, 0, 1, alpha, beta, "P")
ax = fig.add_subplot(131, projection="polar")
ax.plot(np.real(i), np.abs(infspc), label="P-wave")
ax.plot(np.real(i), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/alpha)
ax.set_theta_zero_location("S")
ax.set_rticks([])

# SV-waves
p = np.linspace(-1/beta, 1/beta, 1001)
j = to_incidence(p, beta)
infspc = infinite_space(0, 0, 0, 1, 0, 0, p, 0, 1, beta, "SV", -1)
hlfspc = half_space(0, 0, 0, 1, 0, 0, p, 0, 1, alpha, beta, "SV")
ax = fig.add_subplot(132, projection="polar")
ax.plot(np.real(j), np.abs(infspc), label="SV-wave")
ax.plot(np.real(j), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/beta)
ax.set_theta_zero_location("S")
ax.set_rticks([])

ax.text(
    0.5, 0,
    "Vertical directivity", size=12,
    ha="center", transform=ax.transAxes)

# SH-waves
p = np.linspace(-1/beta, 1/beta, 1001)
j = to_incidence(p, beta)
infspc = 2*infinite_space(0, 0, 0, np.sqrt(2), np.sqrt(2),
                          0, p, 0, 1, beta, "SH", -1)
hlfspc = 2*half_space(0, 0, 0, np.sqrt(2), np.sqrt(2),
                      0, p, 0, 1, alpha, beta, "SH")
ax = fig.add_subplot(133, projection="polar")
ax.plot(np.real(j), np.abs(infspc), label="SH-wave")
ax.plot(np.real(j), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/beta)
ax.set_theta_zero_location("S")
ax.set_rticks([])

fig.tight_layout()

# %%
# Horizontal directivity

alpha = 5
beta = 3

fig = plt.figure(figsize=(10, 4), dpi=144)

# P-waves
d = np.linspace(-np.pi, np.pi, 1001)
infspc = infinite_space(0, 0, 0, np.cos(d)/alpha, np.sin(d) /
                        alpha, 0, 1/alpha, 0, 1, alpha, "P", -1)
ax = fig.add_subplot(131, projection="polar")
ax.plot(d, np.abs(infspc), c="C2")
ax.set_rlim(0, 2/alpha)
ax.set_rticks([])
ax.set_title("P-wave")

# SV-waves
d = np.linspace(-np.pi, np.pi, 1001)
infspc = 2*infinite_space(0, 0, 0, np.cos(d)/beta, np.sin(d) /
                          beta, 0, 1/beta/np.sqrt(2), 0, 1, beta, "SV", -1)
ax = fig.add_subplot(132, projection="polar")
ax.plot(d, np.abs(infspc), c="C2")
ax.set_rlim(0, 2/beta)
ax.set_rticks([])
ax.set_title("SV-wave")

ax.text(
    0.5, -0.2,
    "Horizontal directivity", size=12,
    ha="center", transform=ax.transAxes)

# SH-waves
d = np.linspace(-np.pi, np.pi, 1001)
infspc = infinite_space(0, 0, 0, np.cos(d)/beta, np.sin(d) /
                        beta, 0, 1/beta,  0, 1, beta, "SH", -1)
ax = fig.add_subplot(133, projection="polar")
ax.plot(d, np.abs(infspc), c="C2")
ax.set_rlim(0, 2/beta)
ax.set_rticks([])
ax.set_title("SH-wave")

fig.tight_layout()

# %% Bore-hole fiber

alpha = 5
beta = 3

fig = plt.figure(figsize=(10, 4), dpi=144)

# P-waves z = 0
p = np.linspace(-1/alpha, 1/alpha, 1001)
i = to_incidence(p, alpha)
infspc = infinite_space(0, 0, 0, 0, 0, 1, p, 0, 1, alpha, "P", -1)
hlfspc = half_space(0, 0, 0, 0, 0, 1, p, 0, 1, alpha, beta, "P")
ax = fig.add_subplot(131, projection="polar")
ax.plot(np.real(i), np.abs(infspc), label="P-wave")
ax.plot(np.real(i), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/alpha)
ax.set_theta_zero_location("S")
ax.set_rticks([])
ax.set_title(r"$z = 0$")


# P-waves z = lambda / 4
p = np.linspace(-1/alpha, 1/alpha, 1001)
i = to_incidence(p, alpha)
infspc = infinite_space(0, 0, np.pi*alpha/2, 0, 0, 1, p, 0, 1, alpha, "P", -1)
hlfspc = half_space(0, 0, np.pi*alpha/2, 0, 0, 1, p, 0, 1, alpha, beta, "P")
ax = fig.add_subplot(132, projection="polar")
ax.plot(np.real(i), np.abs(infspc), label="P-wave")
ax.plot(np.real(i), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/alpha)
ax.set_theta_zero_location("S")
ax.set_rticks([])
ax.set_title(r"$z = \lambda/4$")


# P-waves z = lambda / 12
p = np.linspace(-1/alpha, 1/alpha, 1001)
i = to_incidence(p, alpha)
infspc = infinite_space(0, 0, np.pi*alpha/6, 0, 0, 1, p, 0, 1, alpha, "P", -1)
hlfspc = half_space(0, 0, np.pi*alpha/6, 0, 0, 1, p, 0, 1, alpha, beta, "P")
ax = fig.add_subplot(133, projection="polar")
ax.plot(np.real(i), np.abs(infspc), label="Free space")
ax.plot(np.real(i), np.abs(hlfspc), label="Free surface")
ax.legend()
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_rlim(0, 2/alpha)
ax.set_theta_zero_location("S")
ax.set_rticks([])
ax.set_title(r"$z = \lambda/12$")
