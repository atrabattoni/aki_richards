# %%
from sympy import sin, cos, exp, pi, I, symbols
from sympy import diff, Matrix, simplify

# %%
# Starting from Table 5.1 of Aki & Richards.

# Symbols
i, j, d = symbols(r"i, j, \delta")  # angles
a, b, w = symbols(r"\alpha, \beta, \omega")  # waves' speed and pulsation
x, y, z, t = symbols(r"x, y, z, t")  # spatial and temporal dimensions
PP, PS, SP, SS = symbols("PP, PS, SP, SS")  # amplitude coefficients

r = Matrix([x, y, z])  # spatial vector

# Oscillatory terms
opu = exp(I * w * (sin(i) / a * x - cos(i) / a * z - t))
opd = exp(I * w * (sin(i) / a * x + cos(i) / a * z - t))
osu = exp(I * w * (sin(j) / b * x - cos(j) / b * z - t))
osd = exp(I * w * (sin(j) / b * x + cos(j) / b * z - t))

# Unitary vectors
vx = Matrix([1, 0, 0])
vy = Matrix([0, 1, 0])
vz = Matrix([0, 0, 1])
vd = Matrix([cos(d), sin(d), 0])
vpu = Matrix([sin(i), 0, - cos(i)])
vpd = Matrix([sin(i), 0, + cos(i)])
vsvu = Matrix([cos(j), 0, + sin(j)])
vsvd = Matrix([cos(j), 0, - sin(j)])
vh = vy

# Displacement
up = vpu * opu + PP * vpd * opd + PS * vsvd * osd
usv = vsvu * osu + SP * vpd * opd + SS * vsvd * osd
ush = vh * (osu + osd)  # Total reflection for SH-waves

# Strain
ep = Matrix([[(diff(up[i], r[j]) + diff(up[j], r[i])) / 2
              for i in range(3)] for j in range(3)])
esv = Matrix([[(diff(usv[i], r[j]) + diff(usv[j], r[i])) / 2
               for i in range(3)] for j in range(3)])
esh = Matrix([[(diff(ush[i], r[j]) + diff(ush[j], r[i])) / 2
               for i in range(3)] for j in range(3)])

# Strain-rate
edotp = diff(ep, t)
edotsv = diff(esv, t)
edotsh = diff(esh, t)

# %% Horizontal analysis

# Projection along x
subs = {x: 0, y: 0, z: 0, t: 0}
edotxxp, = (vx.T * edotp * vx).subs(subs)
edotxxsv, = (vx.T * edotsv * vx).subs(subs)

# Projection along v(delta)
edotddp, = (vd.T * edotp * vd).subs(subs)
edotddsv, = (vd.T * edotsv * vd).subs(subs)
edotddsh, = (vd.T * edotsh * vd).subs(subs)

# %% Vertical analysis

# Projection along z
subs = {x: 0, y: 0, t: 0}
edotzzp, = (vz.T * edotp * vz).subs(subs)
edotzzsv, = (vz.T * edotsv * vz).subs(subs)
edotzzsh, = (vz.T * edotsh * vz).subs(subs)

# Waves incomming from the z axis
k = symbols("k")
subs = {i: 0, j: 0, PP: -1, PS: 0, SP: 0, SS: 1, w: a*k}
edotzzpz = simplify(edotzzp.subs(subs))
edotzzsvz = simplify(edotzzsv.subs(subs))

# Pattern at the surface
subs = {z: 0}
edotzzp0 = edotzzp.subs(subs)
edotzzsv0 = edotzzsv.subs(subs)

# Pattern at lambda/4
subs = {z: pi * a / 2 / w}
edotzzp0 = edotzzp.subs(subs)
edotzzsv0 = edotzzsv.subs(subs)
