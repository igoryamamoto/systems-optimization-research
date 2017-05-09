# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: igoryamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import ihmpctools as ihmpc

#%% Transfer Functions Gij (output i/ input j)

# g11 = -0.19/s
b11 = np.array([-0.19])
a11 = np.array([1, 0])

# g12 = -1.7/(19.5*s+1)
b12 = np.array([-1.7])
a12 = np.array([19.5, 1])

# g21 = -0.763/(31.8*s+1)
b21 = np.array([-0.763])
a21 = np.array([31.8, 1])

# g22 = 0.235/s
b22 = np.array([0.235])
a22 = np.array([1, 0])

#%% Define parameters

# dt, m, ny, nu, na, D0, Dd, Di, F, N, Z, W, Q, R, r, G1, G2, G3
Ts = 1 # sample time
m = 3 # control horizon
ny = 2 # number of outputs
nu = 2 # number of inputs
na = 1 # max order of Gij
nd = ny*nu*na

# Obtain coefficients of partial fractions of Gij/s
d0_11, dd_11, di_11 = ihmpc.get_coeff(b11, a11)
d0_12, dd_12, di_12 = ihmpc.get_coeff(b12, a12)
d0_21, dd_21, di_21 = ihmpc.get_coeff(b21, a21)
d0_22, dd_22, di_22 = ihmpc.get_coeff(b22, a22)

#TODO: Define matrices D0[nyxnu]. Dd[ndxnd], Di[nyxnu] for each Gij
D0 = np.vstack((np.hstack((d0_11, d0_12)),
               np.hstack((d0_21, d0_22))))
Di =
Dd =
#TODO: Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]

#TODO: Define matrix Wn[ndxnd.m] from F

#TODO: Define matrix Z[?] from Dd,N

#TODO: Define matrices D0n[nyxm.nu], Di1n[nyxm.nu], D12n[nyxm.nu]

#TODO: Obtain G1, G2, G3

#TODO:
























#%%
t_sim = 300

# Real Process
Br11 = np.array([-0.19])
Ar11 = np.array([1, -1])
Br12 = np.array([-0.08498])
Ar12 = np.array([1, -0.95])
Br21 = np.array([-0.02362])
Ar21 = np.array([1, -0.969])
Br22 = np.array([0.235])
Ar22 = np.array([1, -1])

na11 = len(Ar11)
na12 = len(Ar12)
na21 = len(Ar21)
na22 = len(Ar22)
nb = 1

#%% Set point and Disturbance Signals
w1 = np.array([1]*(t_sim))
w2 = np.array([1]*(t_sim))

#%% Initialization
y11 = np.zeros(t_sim+1)
y12 = np.zeros(t_sim+1)
y21 = np.zeros(t_sim+1)
y22 = np.zeros(t_sim+1)
u1 = np.zeros(t_sim+1)
u2 = np.zeros(t_sim+1)
du1 = np.zeros(t_sim+1)
du2 = np.zeros(t_sim+1)
y11_past = np.zeros(na11)
y12_past = np.zeros(na12)
y21_past = np.zeros(na21)
y22_past = np.zeros(na22)
u1_past = np.zeros(nb)
u2_past = np.zeros(nb)

#%% Control Loop
for k in range(1,t_sim+1):
    # Real process
    y11[k] = -Ar11[1:].dot(y11_past[:-1]) + Br11.dot(u1_past)
    y12[k] = -Ar12[1:].dot(y12_past[:-1]) + Br12.dot(u2_past)
    y21[k] = -Ar21[1:].dot(y21_past[:-1]) + Br21.dot(u1_past)
    y22[k] = -Ar22[1:].dot(y22_past[:-1]) + Br22.dot(u2_past)

    # Free Response


    # Select set points for the current horizon
    w = []
    # Solver Inputs
    H = matrix()
    q = matrix()
    A = matrix()
    b = matrix()
    # Solve
    sol = solvers.qp(P=H,q=q,G=A,h=b)
    dup = list(sol['x'])

    du1[k] = dup[0]
    du2[k] = dup[m]
    u1[k] = u1[k-1] + du1[k]
    u2[k] = u2[k-1] + du2[k]

    u1_past = np.append(u1[k],u1_past[:-1])
    u2_past = np.append(u2[k],u2_past[:-1])
    y11_past = np.append(y11[k],y11_past[:-1])
    y12_past = np.append(y12[k],y12_past[:-1])
    y21_past = np.append(y21[k],y21_past[:-1])
    y22_past = np.append(y22[k],y22_past[:-1])


#%% Teste
plt.clf()
plt.plot([1]*(t_sim+1),':', label='Target')
plt.plot(y11+y12, label='y1')
plt.plot(y21+y22, label='y2')
plt.plot(u1,'--', label='u1')
plt.plot(u2,'--', label='u2')
plt.legend(loc=4)
plt.xlabel('sample time (k)')
