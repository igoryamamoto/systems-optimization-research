# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:58:03 2017

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
    
# prediction horizon
p = 15
# control horizon
m = 3
# number of inputs
nu = 2
# number of outputs
ny = 2

#%% vectors with step responses till t=15s w/ T = 1s
# gij -> output i relative to input j
# tf = -0.19/s
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

#%% Create Dynamic Matrix
G11 = create_matrix_G(g11,p,m)
G12 = create_matrix_G(g12,p,m)
G21 = create_matrix_G(g21,p,m)
G22 = create_matrix_G(g22,p,m)   
G1 = np.hstack((G11,G12))
G2 = np.hstack((G21,G22))
G = np.vstack((G1,G2))

#%% Initialization
# Number of sample points to simulate
num_samples = 150
# References
w1 = np.array([1]*(num_samples+p))
w2 = np.array([1]*(num_samples+p))
# weights
Q = np.diag(np.ones(p*ny))
R = np.diag([10**1]*m*nu)
# output and input vectors to store computed values
y11 = y12 = y21 = y22 = u1 = u2 = np.zeros(num_samples+1)
du1 = du2 = np.zeros(num_samples)
N=150
vect_g11 = vect_g12 = vect_g21 = vect_g22 = np.zeros(N)
# dux_f= [dux(k-1) dux(k-2) ... dux(k-N)]
du1_f = du2_f = np.zeros(N)
# Free respose
f = np.zeros(p*ny)
dumax = 0.2

#%% Control Loop
for i in range(num_samples):
    # Process Output (equal model in this case)
    y11[i+1] = y11[i] - 0.19*u1[i]
    y12[i+1] = 0.95*y12[i] - 0.08498*u2[i]
    y21[i+1] = 0.969*y21[i] - 0.02362*u1[i]
    y22[i+1] = y22[i] + 0.235*u2[i]
    for k in range(p):
        for j in range(N):
            vect_g11[j] = g11[k+j]-g11[j]
            vect_g12[j] = g12[k+j]-g12[j]
            vect_g21[j] = g21[k+j]-g21[j]
            vect_g22[j] = g22[k+j]-g22[j]
        f[k] = y11[i+1] + y12[i+1] + vect_g11.dot(du1_f) + vect_g12.dot(du2_f)
        f[p+k] = y21[i+1] + y22[i+1] + vect_g21.dot(du1_f) + vect_g22.dot(du2_f)
    # Select references for the current horizon
    w = np.append(w1[i:i+p], w2[i:i+p])
    # solver inputs
    H = matrix((2*np.transpose(G).dot(Q).dot(G)+R).tolist())
    q = matrix((2*np.transpose(G).dot(Q).dot(f-w)).tolist())
    A = matrix(np.diag(np.ones(nu*m)).tolist()+np.diag(-1*np.ones(nu*m)).tolist())
    b = matrix([dumax]*2*nu*m)
    # solve
    sol = solvers.qp(P=H,q=q,G=A.T,h=b)
    dup = list(sol['x'])
    #print(du)
    du1[i] = dup[0]
    du2[i] = dup[m]
    u1[i+1] = u1[i] + du1[i]
    u2[i+1] = u2[i] + du2[i]
    
    du1_f = np.append(du1[i], du1_f[:-1])
    du2_f = np.append(du2[i], du2_f[:-1])  
    
#%% Plot
#du1 = np.array(du[0:m])
#du2 = np.array(du[m:])
#x11 = G11.dot(du1)
#x12 = G12.dot(du2)
#x21 = G21.dot(du1)
#x22 = G22.dot(du2)
#y1 = np.append(np.zeros(1),x11+x12)
#y2 = np.append(np.zeros(1),x21+x22)
#u1 = u2 = np.array([0])
#for i in range(m):
#    u1 = np.append(u1,u1[i]+du1[i])
#    u2 = np.append(u2,u2[i]+du2[i])
#u1 = np.append(u1,[u1[m]]*(p-m))
#u2 = np.append(u2,[u2[m]]*(p-m))
plt.clf()
plt.plot(w1[:num_samples+1],':', label='Reference')
plt.plot(y11+y12, label='y1')
plt.plot(y21+y22, label='y2')
plt.plot(u1,'--', label='u1')
plt.plot(u2,'--', label='u2')
plt.legend(loc=4)
plt.xlabel('sample time (k)')
