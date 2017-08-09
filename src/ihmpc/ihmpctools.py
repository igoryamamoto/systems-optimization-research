# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:46:44 2017

@author: peixeboi
"""
import numpy as np
from scipy import signal

def get_coeff(b,a):
    # multiply by 1/s (step)
    a = np.append(a, 0)
    # do partial fraction expansion
    r,p,k = signal.residue(b,a)
    # r: Residues
    # p: Poles
    # k: Coefficients of the direct polynomial term

    d_s = np.array([])
    d_d = np.array([])
    d_i = np.array([])
    poles = np.array([])
    integrador = 0

    for i in range(np.size(p)):
        if (p[i] == 0):
            if (integrador):
                d_i = np.append(d_i, r[i])
            else:
                d_s = np.append(d_s, r[i])
                integrador += 1
        else:
            d_d = np.append(d_d, r[i])
            poles = np.append(poles, p[i])

    if (d_i.size == 0):
        d_i = np.append(d_i, 0)

    return d_s, d_d, d_i

b12 = np.array([-0.19])
a12 = np.array([212, 1])
ds, dd, di = get_coeff(b12,a12)
print(ds)
print(dd)
print(di)