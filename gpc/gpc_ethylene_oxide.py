# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:34:03 2017

@author: Ígor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def create_matrix_G(g, p, m):
    g = np.append(g[:p],np.zeros(m-1))
    G = np.array([])
    for i in range(p):
        G = np.append(G,[g[i-j] for j in range(m)])  
    return np.resize(G,(p,m))

#%% Simulation Parameters
t_sim = 200

#%% Controller Parameters
p = 15    # prediction horizon 
m = 3     # control horizon
nu = 2    # number of inputs
ny = 2    # number of outputs
Q = np.eye(p*ny)
R = 10**-2*np.eye(m*nu)
du_max = 0.2
du_min = -0.2

#%% Create Matrix G    
g11 = open('g11','r').read().splitlines()
g11 = np.array(list(map(float,g11)))
# tf = -1.7/(19.5*s+1) 
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

# Process Model
Bm11 = np.array([-0.19])
Am11 = np.array([1 -1])
Bm12 = np.array([-0.08498])
Am12 = np.array([1 -0.95])
Bm21 = np.array([-0.02362])
Am21 = np.array([1 -0.969])
Bm22 = np.array([0.235])
Am22 = np.array([1 -1])

# Real Process
Br11 = np.array([-0.19])
Ar11 = np.array([1 -1])
Br12 = np.array([-0.08498])
Ar12 = np.array([1 -0.95])
Br21 = np.array([-0.02362])
Ar21 = np.array([1 -0.969])
Br22 = np.array([0.235])
Ar22 = np.array([1 -1])

#%% Reference and Disturbance Signals
w1 = np.array([1]*(t_sim+p))
w2 = np.array([1]*(t_sim+p))

#%% Initialization
y11 = y12 = y21 = y22 = u1 = u2 = np.zeros(t_sim+1)
du1 = du2 = np.zeros(t_sim+1)
y11_past = np.zeros(len(Ar11))
y12_past = np.zeros(len(Ar12))
y21_past = np.zeros(len(Ar21))
y22_past = np.zeros(len(Ar22))
u1_past = np.zeros(1)
u2_past = np.zeros(1)

#%% Control Loop
for k in range(1,t_sim+1):
    y11[k] = Ar11.dot(y11_past) + Br11.dot(u1_past)
    y12[k] = Ar12.dot(y12_past) + Br12.dot(u2_past)
    y21[k] = Ar21.dot(y21_past) + Br21.dot(u1_past)
    y22[k] = Ar22.dot(y22_past) + Br22.dot(u2_past)
    
    # Free Response
    #??
    f = np.zeros(ny*p)
    # Select references for the current horizon
    w = np.append(w1[k:k+p], w2[k:k+p])
    # Solver Inputs
    H = matrix((2*(np.transpose(G).dot(Q).dot(G)+R)).tolist())
    q = matrix((2*np.transpose(G).dot(Q).dot(f-w)).tolist())
    A = matrix(np.hstack((np.eye(nu*m),-1*np.eye(nu*m))).tolist())
    b = matrix([du_max]*nu*m+[-du_min]*nu*m)
    # Solve
    sol = solvers.qp(P=H,q=q,G=A,h=b)
    dup = list(sol['x'])
    
    du1[k] = dup[0]
    du2[k] = dup[m]
    u1[k] = u1[k-1] + du1[k]
    u2[k] = u2[k-1] + du2[k]
    
    y11_past = np.append(y11[k],y11_past[:-1])
    y12_past = np.append(y12[k],y12_past[:-1])
    y21_past = np.append(y21[k],y21_past[:-1])
    y22_past = np.append(y22[k],y22_past[:-1])


#%% Teste
x11 = G11.dot(dup[:m])
x12 = G12.dot(dup[m:])
x21 = G21.dot(dup[:m])
x22 = G22.dot(dup[m:])
y1 = np.append(np.zeros(1),x11+x12)
y2 = np.append(np.zeros(1),x21+x22)
u1 = u2 = np.array([0])
for i in range(m):
    u1 = np.append(u1,u1[i]+dup[i])
    u2 = np.append(u2,u2[i]+dup[m+i])
u1 = np.append(u1,[u1[m]]*(p-m))
u2 = np.append(u2,[u2[m]]*(p-m))
plt.clf()
plt.plot([1]*p,':', label='Reference')
plt.plot(y1, label='y1')
plt.plot(y2, label='y2')
plt.plot(u1,'--', label='u1')
plt.plot(u2,'--', label='u2')
plt.legend(loc=4)
plt.xlabel('sample time (k)')





