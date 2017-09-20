#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:05:34 2017

@author: peixeboi
"""
import numpy as np
def psi(t):
    ny = 2
    nd = 4
    nu = 2
    na = 1
    R = np.array([np.array([ 0.]), np.array([-0.05128205]), np.array([-0.03144654]), np.array([ 0.])])
    R2 = np.array(list(map(lambda x: np.exp(x*t), R)))
    psi = np.zeros((ny, nd))
    for i in range(ny):
        phi = np.array([])
        for j in range(nu):
            phi = np.concatenate((phi, R2[i*nu+j]))
        psi[i, i*nu*na:(i+1)*nu*na] = phi      
    return psi

def G1(n):
    ny = 2
    nd = 4
    nu = 2
    na = 1
    R = np.array([np.array([ 0.]), np.array([-0.05128205]), np.array([-0.03144654]), np.array([ 0.])])
    G = np.zeros((ny, nd))
    for i in range(ny):
        phi = np.array([])
        for j in range(nu):
            r = R[i*nu+j]
            if r == 0:
                g = n
            else:
                g = 1/r*(np.exp(r*n)-1)
            phi = np.append(phi, g)
        G[i, i*nu*na:(i+1)*nu*na] = phi      
    return G

from scipy.linalg import block_diag
def G2(n):
    nu = 2
    ny = 2
    R = np.array([np.array([ 0.]), np.array([-0.05128205]), np.array([-0.03144654]), np.array([ 0.])])
    G = np.array([])
    for y in range(ny):
        r = R[y*nu:y*nu+nu]
        g = np.zeros((nu, nu))
        for i in range(nu):
            for j in range(nu):
                g[i, j] = r[i] + r[j]
        G = block_diag(G, g)
    return G[1:]
        
def G3(n):
    ny = 2
    nd = 4
    nu = 2
    na = 1
    R = np.array([np.array([ 0.]), np.array([-0.05128205]), np.array([-0.03144654]), np.array([ 0.])])
    G = np.zeros((ny, nd))
    for i in range(ny):
        phi = np.array([])
        for j in range(nu):
            r = R[i*nu+j]
            phi = np.append(phi, r)
        if 0 in phi:
            def aux(x):
                if x==0:
                    return 0
                else:
                    return (1/x**2)*np.exp(x*n)*(x*n-1)
            phi = np.array(list(map(aux, phi)))
        else:
            phi = np.zeros(nu)
        G[i, i*nu*na:(i+1)*nu*na] = phi 
    return G

H_m = 0
for n in range(m):
    a = Z.T.dot(Wn[n].T).dot(G2(n, R)-G2(n-1, R)).dot(Wn[n]).dot(Z)
    b1 = (G1(n) - G1(n-1)).T.dot(Q).dot(D0_n[n]-Di_2n[n])
    b2 = (G3(n, R)-G3(n-1, R)).T.dot(Q).dot(Di_1n[n])
    b3 = (G1(n, R)-G1(n-1, R)).T.dot(Q).dot(Di_1n[n])
    b = 2*Z.T.dot(Wn[n].T).dot(b1+b2+b3)
    c1 = Ts*(D0_n[n]-Di_2n[n]).T.dot(Q).dot(D0_n[n]-Di_2n[n])
    c2 = 2*(n-1/2)*Ts**2*(D0_n[n]-Di_2n[n]).T.dot(Q).dot(Di_1n[n])
    c3 = (n**3-n+1/3)*Ts**3*Di_1n[n].T.dot(Q).dot(Di_1n[n])
    c = c1 + c2 + c3
    H_m += a + b + c

H_inf = Z.T.dot(W_m.T).dot(G2_inf-G2_m).dot(W_m).dot(Z)

H = H_m + H_inf

cf_m = 0
for n in range(m):
    a = (-e_s.T.dot(Q).dot(G1(n, R)-G1(n-1, R)) + x_d.T * (G2(n, R)-G2(n-1, R)) + x_i.T.dot(Q.dot(G3(n, R)-G3(n-1, R))))*Wn[n]*Z
    b = (-Ts*e_s.T + (n-1/2)*Ts**2*x_i.T+x_d.T*(G1(n, R)-G1(n-1, R)).T).dot(Q).dot(D0_n[n]-Di_2n[n])
    c = (-(n-1/2)*Ts**2*e_s.T + (n**3-n+1/3)*Ts**3*x_i.T + x_d.T*(G3(n, R)-G3(n-1, R)).T).dot(Q).dot(Di_1n[n])
    cf_m += a + b + c 

cf_inf = x_d.T*(G2_inf-G2_m).dot(W_m).dot(Z)

cf = cf_m + cf_inf