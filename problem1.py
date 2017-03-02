# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:58:03 2017

@author: Ígor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def create_matrix_G(g, p, m):
    g = np.append(g,np.zeros(m-1))
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
g11 = np.array([
    -0.190000000000000,
    -0.380000000000000,
    -0.570000000000000,
    -0.760000000000000,
    -0.950000000000000,
    -1.14000000000000,
    -1.33000000000000,
    -1.52000000000000, 
    -1.71000000000000,
    -1.90000000000000,
    -2.09000000000000,
    -2.28000000000000,
    -2.47000000000000,
    -2.66000000000000,
    -2.85000000000000
])
# tf = -0.763/(31.8*s+1)
g12 = np.array([
    -0.0236203746985634,
    -0.0465095277707828,
    -0.0686900958205925,
    -0.0901840146851289,
    -0.111012541128537,
    -0.131196273864197,
    -0.150755173926158,
    -0.169708584409933,
    -0.188075249602173,
    -0.205873333518129,
    -0.223120437865263,
    -0.239833619450740,
    -0.256029407050039,
    -0.271723817753356,
    -0.286932372805971
])
# tf = -1.7/(19.5*s+1) 
g21 = np.array([
    -0.0849818422825443,
    -0.165715500142891,
    -0.242413337427250,
    -0.315277102057778,
    -0.384498456715398,
    -0.450259482994136,
    -0.512733160353135,
    -0.572083821126173,
    -0.628467582785576,
    -0.682032758597547,
    -0.732920247749111,
    -0.781263905972892,
    -0.827190897644593,
    -0.870822030279379,
    -0.912272072307002
])
# tf = 0.235/s
g22 = np.array([
    0.235000000000000,
    0.470000000000000,
    0.705000000000000,
    0.940000000000000,
    1.17500000000000,
    1.41000000000000,
    1.64500000000000,
    1.88000000000000,
    2.11500000000000,
    2.35000000000000,
    2.58500000000000,
    2.82000000000000,
    3.05500000000000,
    3.29000000000000,
    3.52500000000000
])
#%% Create Dynamic Matrix
G11 = create_matrix_G(g11,p,m)
G12 = create_matrix_G(g12,p,m)
G21 = create_matrix_G(g21,p,m)
G22 = create_matrix_G(g22,p,m)   
G1 = np.hstack((G11,G12))
G2 = np.hstack((G21,G22))
G = np.vstack((G1,G2))
# Free respose
f = np.zeros(p*ny)
# References
w = np.array([1]*p+[1]*p)
# weights
Q = np.diag(np.ones(p*ny))
R = np.diag([10**-2]*m*nu)
#%% solver inputs
H = matrix((2*np.transpose(G).dot(Q).dot(G)+R).tolist())
q = matrix((2*np.transpose(G).dot(Q).dot(f-w)).tolist())
A = matrix(np.diag(np.ones(nu*m)).tolist()+np.diag(-1*np.ones(nu*m)).tolist())
b = matrix([0.2]*2*nu*m)
# solve
sol = solvers.qp(P=H,q=q,G=A.T,h=b)
sol_x = sol['x']
print(sol_x)
#%%
du1 = np.array([sol_x[i] for i in range(m)])
du2 = np.array([sol_x[i+m] for i in range(m)])
x11 = G11.dot(du1)
x12 = G12.dot(du2)
x21 = G21.dot(du1)
x22 = G22.dot(du2)
plt.plot(x11+x12)
plt.plot(x21+x22)
plt.legend(['y1','y2'])