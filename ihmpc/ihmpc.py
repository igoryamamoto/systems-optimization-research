# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: igoryamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy import signal

def get_coeff(b,a,na):
    # multiply by 1/s (step)
    a = np.append(a, 0)
    # do partial fraction expansion
    r,p,k = signal.residue(b,a)
    # r: Residues
    # p: Poles
    # k: Coefficients of the direct polynomial term

    d_s = np.array([])
    #d_d = np.array([])
    d_d = np.zeros(na)
    d_i = np.array([])
    poles = np.array([])
    integrador = 0
    i = 0;

    for i in range(np.size(p)):
        if (p[i] == 0):
            if (integrador):
                d_i = np.append(d_i, r[i])
            else:
                d_s = np.append(d_s, r[i])
                integrador += 1
        else:
            d_d[i] = r[i]
            poles = np.append(poles, p[i])
            i += 1

    if (d_i.size == 0):
        d_i = np.append(d_i, 0)

    return d_s, d_d, d_i, poles

def create_J(nu, na):
    J = np.zeros((nu*na, nu))
    for col in range(nu):
        J[col*na:col*na+na, col] = np.ones(na)
    return J

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
d0_11, dd_11, di_11, r_11 = get_coeff(b11, a11,na)
d0_12, dd_12, di_12, r_12 = get_coeff(b12, a12,na)
d0_21, dd_21, di_21, r_21 = get_coeff(b21, a21,na)
d0_22, dd_22, di_22, r_22 = get_coeff(b22, a22,na)

# Define matrices D0[nyxnu]. Dd[ndxnd], Di[nyxnu] for each Gij
D0 = np.vstack((np.hstack((d0_11, d0_12)),
               np.hstack((d0_21, d0_22))))
Di = np.vstack((np.hstack((di_11, di_12)),
               np.hstack((di_21, di_22))))
Dd = np.diag(dd_11.tolist() + dd_11.tolist() + dd_21.tolist() + dd_22.tolist());

#TODO: Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]
J = create_J(nu, na)
#FIXME: na changes according to Gij

#N =

F = np.diag(np.exp(r_11.tolist() + r_12.tolist() + r_21.tolist() + r_22.tolist()))

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
