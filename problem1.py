# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:58:03 2017

@author: Ígor Yamamoto
"""
import numpy as np
from cvxopt import matrix, solvers

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
    -2.85000000000000,
    0,
    0
])
 
g12 = np.array([
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
    -0.912272072307002,
    0,
    0
])

g21 = np.array([
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
    -0.286932372805971,
    0,
    0
])

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
    3.52500000000000,
    0,
    0
])
G11 = G12 = G21 = G22 = np.array([0,0,0])
for i in range(np.size(g11)-2):
    row = np.array([g11[i], g11[i-1], g11[i-2]])
    G11 = np.vstack((G11,row))
    
    row = np.array([g12[i], g12[i-1], g12[i-2]])
    G12 = np.vstack((G12,row))
    
    row = np.array([g21[i], g21[i-1], g21[i-2]])
    G21 = np.vstack((G21,row))
    
    row = np.array([g22[i], g22[i-1], g22[i-2]])
    G22 = np.vstack((G22,row))
    
G11 = np.delete(G11,0,0)
G12 = np.delete(G12,0,0)
G21 = np.delete(G21,0,0)
G22 = np.delete(G22,0,0)    
    
G1 = np.hstack((G11,G12))
G2 = np.hstack((G21,G22))
G = np.vstack((G1,G2))






