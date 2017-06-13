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
    poles = np.zeros(na)
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
            poles[i] = p[i]
            i += 1

    if (d_i.size == 0):
        d_i = np.append(d_i, 0)

    return d_s, d_d, d_i, poles

def create_J():
    J = np.zeros((nu*na, nu))
    for col in range(nu):
        J[col*na:col*na+na, col] = np.ones(na)
    return J

def create_psi():
    Psi = np.zeros((ny, nd))
    for row in range(ny):
        Psi[row, row*(nu*na):row*(nu*na)+nu*na] = np.ones(nu*na)
    return Psi

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
nx = 2*ny+nd

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
Dd = np.diag(dd_11.tolist() + dd_12.tolist() + dd_21.tolist() + dd_22.tolist());

# Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]
J = create_J()

F = np.diag(np.exp(r_11.tolist() + r_12.tolist() + r_21.tolist() + r_22.tolist()))

N = np.vstack((J, J))

a1 = np.hstack((np.eye(ny), np.zeros((ny,nd)), Ts*np.eye(ny)))
a2 = np.hstack((np.zeros((nd,ny)), F, np.zeros((nd,ny))))
a3 = np.hstack((np.zeros((ny,ny)), np.zeros((ny,nd)), np.eye(ny)))
A = np.vstack((a1, a2, a3))

B = np.vstack((D0+Ts*Di, Dd.dot(F).dot(N), Di))

Psi = create_psi()
C = np.hstack(( np.eye(ny), Psi, np.zeros((ny,ny)) ))

D = np.zeros((ny,nu))
ethylene = signal.StateSpace(A, B, C, D)

tsim = 100
U = np.vstack(( [1,1] ,np.zeros((tsim-1,2)) ))
T = np.arange(tsim)
#res = ethylene.output(U, T)
#plt.plot(res[1])
X = np.zeros((tsim,8))
Y = np.zeros((tsim,2))
for k in T[:-1]:
    X[k+1] = A.dot(X[k]) + B.dot(U[k])
    Y[k+1] = C.dot(X[k])
plt.plot(Y)
#%%

class SystemModel(object):
    def __init__(self, ny, nu, *args):
        self.ny = ny
        self.nu = nu
        self.H = list(*args)

    def step_response(self, X0=None, T=None, N=None):
        def fun(X02=None, T2=None, N2=None):
            def fun2(sys):
                return signal.step(sys, X02, T2, N2)
            return fun2
        fun3 = fun(X0, T, N)
        step_with_time = list(map(fun3, self.H))
        return [s[1] for s in step_with_time]
    
h11 = signal.TransferFunction([-0.19],[1, 0])
h12 = signal.TransferFunction([-1.7],[19.5, 1])
h21 = signal.TransferFunction([-0.763],[31.8, 1])
h22 = signal.TransferFunction([0.235],[1, 0])
ethylene2 = SystemModel(2, 2, [h11, h12, h21, h22])
g11, g12, g21, g22 = ethylene2.step_response(T=T)
res2 = np.vstack((g11+g12, g21+g22))
plt.plot(res2.T)