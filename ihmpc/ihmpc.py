# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: igoryamamoto
"""
import numpy as np

# dt, m, ny, nu, na, D0, Dd, Di, F, N, Z, W, Q, R, r, G1, G2, G3
Ts = 1 # sample time
m = 3 # control horizon
ny = 2 # number of outputs
nu = 2 # number of inputs
na = 1 # max order of Gij
nd = ny*nu*na

#%% Transfer Functions

# -0.19/s
b11 = np.array([-0.19])
a11 = np.array([1, 0])

# tf = -1.7/(19.5*s+1)
b12 = np.array([-1.7])
a12 = np.array([19.5, 1])

# tf = -0.763/(31.8*s+1)
b21 = np.array([-0.763])
a21 = np.array([31.8, 1])

# tf = 0.235/s
b22 = np.array([0.235])
a22 = np.array([1, 0])

#%%

