# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:03 2017

@author: Ígor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from cvxopt import matrix, solvers

def create_matrix_G(g, p, m):
    g = np.append(g[:p],np.zeros(m-1))
    G = np.array([])
    for i in range(p):
        G = np.append(G,[g[i-j] for j in range(m)])
    return np.resize(G,(p,m))

#%% Simulation Parameters
t_sim = 100

#%% Controller Parameters
p = 15    # prediction horizon
m = 3     # control horizon
nu = 2    # number of inputs
ny = 2    # number of outputs
Q = np.eye(p*ny)
R = 10**1*np.eye(m*nu)
du_max = 0.2
du_min = -0.2

#%% Create Matrix G
# tf = -1.7/(19.5*s+1)
h11 = signal.TransferFunction([-0.19],[1, 0])
h21 = signal.TransferFunction([-1.7],[19.5, 0])
h12 = signal.TransferFunction([-0.763],[31.8, 1])
h22 = signal.TransferFunction([0.235],[1, 0])

#%%
g11 = open('g11','r').read().splitlines()
g11 = np.array(list(map(float,g11)))

g12 = open('g12','r').read().splitlines()
g12 = np.array(list(map(float,g12)))
# tf = -0.763/(31.8*s+1)
g21 = open('g21','r').read().splitlines()
g21 = np.array(list(map(float,g21)))
# tf = 0.235/s
g22 = open('g22','r').read().splitlines()
g22 = np.array(list(map(float,g22)))

G11 = create_matrix_G(g11,p,m)
G12 = create_matrix_G(g12,p,m)
G21 = create_matrix_G(g21,p,m)
G22 = create_matrix_G(g22,p,m)
G1 = np.hstack((G11,G12))
G2 = np.hstack((G21,G22))
G = np.vstack((G1,G2))

#%% Coefficients
dm = 0
dr = 0
# Process Model
Bm11 = np.array([-0.19])
Am11 = np.array([1, -1])
Am11_til = np.convolve(Am11, [1,-1])

Bm12 = np.array([-0.08498])
Am12 = np.array([1, -0.95])
Am12_til = np.convolve(Am12, [1,-1])

Bm21 = np.array([-0.02362])
Am21 = np.array([1, -0.969])
Am21_til = np.convolve(Am21, [1,-1])

Bm22 = np.array([0.235])
Am22 = np.array([1, -1])
Am22_til = np.convolve(Am22, [1,-1])

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

#%% Reference and Disturbance Signals
w1 = np.array([1]*(t_sim+p))
w2 = np.array([1]*(t_sim+p))

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
y11_f = np.zeros(p)
y12_f = np.zeros(p)
y21_f = np.zeros(p)
y22_f = np.zeros(p)

#%% Control Loop
for k in range(1,t_sim+1):
    y11[k] = -Ar11[1:].dot(y11_past[:-1]) + Br11.dot(u1_past)
    y12[k] = -Ar12[1:].dot(y12_past[:-1]) + Br12.dot(u2_past)
    y21[k] = -Ar21[1:].dot(y21_past[:-1]) + Br21.dot(u1_past)
    y22[k] = -Ar22[1:].dot(y22_past[:-1]) + Br22.dot(u2_past)

    # Free Response
    du1_f = np.array(du1[k-1])
    du2_f = np.array(du2[k-1])
    y11_aux = y11_past
    y12_aux = y12_past
    y21_aux = y21_past
    y22_aux = y22_past
    for j in range(p):
        if j <= dm:
            du1_f = np.array(du1[k])
            du2_f = np.array(du2[k])
        else:
            du1_f = du2_f = np.array(0)
        y11_f[j] = -y11_aux.dot(Am11_til[1:]) + du1_f.dot(Bm11)
        y12_f[j] = -y12_aux.dot(Am12_til[1:]) + du2_f.dot(Bm12)
        y21_f[j] = -y21_aux.dot(Am21_til[1:]) + du1_f.dot(Bm21)
        y22_f[j] = -y22_aux.dot(Am22_til[1:]) + du2_f.dot(Bm22)
        y11_aux = np.append(y11_f[j], y11_aux[:-1])
        y12_aux = np.append(y12_f[j], y12_aux[:-1])
        y21_aux = np.append(y21_f[j], y21_aux[:-1])
        y22_aux = np.append(y22_f[j], y22_aux[:-1])
    f = np.append(y11_f+y12_f, y21_f+y22_f)

    # Select references for the current horizon
    w = np.append(w1[k:k+p], w2[k:k+p])
    # Solver Inputs
    H = matrix((2*(G.T.dot(Q).dot(G)+R)).tolist())
    q = matrix((2*G.T.dot(Q).dot(f-w)).tolist())
    A = matrix(np.hstack((np.eye(nu*m),-1*np.eye(nu*m))).tolist())
    b = matrix([du_max]*nu*m+[-du_min]*nu*m)
    # Solve
    sol = solvers.qp(P=H,q=q,G=A,h=b)
    dup = list(sol['x'])
    #dup = np.linalg.inv(G.T.dot(Q).dot(G)+R).dot(G.T).dot(Q).dot(w-f)

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
plt.plot([1]*(t_sim+1),':', label='Reference')
plt.plot(y11+y12, label='y1')
plt.plot(y21+y22, label='y2')
plt.plot(u1,'--', label='u1')
plt.plot(u2,'--', label='u2')
plt.legend(loc=4)
plt.xlabel('sample time (k)')





