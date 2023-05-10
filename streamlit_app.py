# %%
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from aki_richards import scattering_matrix, interface_motion

# %%
t0 = st.write(
    """
    # DAS response

    Sensitivity to incoming P- and S-waves coming from below.
    """
)
left_column, right_column = st.columns(2)
t1 = left_column.write("## Top layer")
r1 = left_column.slider(label=r"Density $\rho_1$", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
a1 = left_column.slider(label=r"P-wave speed $\alpha_1$ [km/s]", min_value=0.0, max_value=10.0, step=0.1, value=1.5)
b1 = left_column.slider(label=r"S-wave speed $\beta_1$ [km/s]", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
t1 = right_column.write("## Bottom layer")
r2 = right_column.slider(label=r"Density $\rho_2$", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
a2 = right_column.slider(label=r"P-wave speed $\alpha_2$ [km/s]", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
b2 = right_column.slider(label=r"S-wave speed $\beta_2$ [km/s]", min_value=0.0, max_value=10.0, step=0.1, value=0.3)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection="polar")

theta = np.linspace(-np.pi / 2, np.pi / 2, 202)[1:-1]

p = np.sin(theta) / a1
out = np.zeros(p.shape, dtype="complex")
for idx, value in enumerate(p):
    I, S = scattering_matrix(r1, a1, b1, r2, a2, b2, value)
    out[idx] = interface_motion(I, S)[0]
ax.plot(theta + np.pi, np.abs(p * out) * a2, label="P")

p = np.sin(theta) / b1
out = np.zeros(p.shape, dtype="complex")
for idx, value in enumerate(p):
    I, S = scattering_matrix(r1, a1, b1, r2, a2, b2, value)
    out[idx] = interface_motion(I, S)[1]
ax.plot(theta + np.pi, np.abs(p * out) * b2, label="S")

p = np.sin(theta) / a2
out = np.zeros(p.shape, dtype="complex")
for idx, value in enumerate(p):
    I, S = scattering_matrix(r1, a1, b1, r2, a2, b2, value)
    out[idx] = interface_motion(I, S)[2]
ax.plot(theta, np.abs(p * out) * a2, label="P")

p = np.sin(theta) / b2
out = np.zeros(p.shape, dtype="complex")
for idx, value in enumerate(p):
    I, S = scattering_matrix(r1, a1, b1, r2, a2, b2, value)
    out[idx] = interface_motion(I, S)[3]
ax.plot(theta, np.abs(p * out) * b2, label="S")

ax.legend()
ax.set_rlim(0, 2)
ax.set_theta_zero_location("S")
ax.set_rticks([0, 0.5, 1, 1.5, 2])

st.write(fig)


# %%
