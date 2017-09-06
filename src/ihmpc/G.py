#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:05:34 2017

@author: peixeboi
"""
import numpy as np
# contar o numero de elementos de R -> problema quando tem mais de um polo numa FT
def G1(n, R):
    nu = 2
    ny = 2
    ns = nu*ny # numero de sistemas -> cuidar que nem sempre sera SISO
    nd = 4 # contar numero de elementos de R
    G = np.zeros((ns, nd))
    count = 0
    for i, l in enumerate(R):
        for r in l:
            if r == 0:
                g = n
            else:
                g = 1/r*(np.exp(r*n)-1)
            G[i, count] = g
            count += 1
    return G

def G2(n, R):
    nu = 2
    ny = 2
    ns = nu*ny # numero de sistemas -> cuidar que nem sempre sera SISO
    nd = 4 # contar numero de elementos de R
    G = np.zeros((ns, nd))
    count = 0
    for i, l in enumerate(R):
        for r in l:
            if r == 0:
                g = n
            else:
                g = 1/(2*r)*(np.exp(2*r*n)-1)
            G[i, count] = g
            count += 1
    return G

def G3(n, R):
    nu = 2
    ny = 2
    ns = nu*ny # numero de sistemas -> cuidar que nem sempre sera SISO
    nd = 4 # contar numero de elementos de R
    G = np.zeros((ns, nd))
    count = 0
    for i, l in enumerate(R):
        for r in l:
            if r == 0:
                g = 1/2*n**2
            else:
                g = 0
            G[i, count] = g
            count += 1
    return G

H_m = 0
for n in range(m):
    a = Z.T.dot(Wn[n].T).dot(G2(n, R)-G2(n-1, R)).dot(Wn[n]).dot(Z)
    b1 = (G1(n, R) - G1(n-1, R)).T.dot(Q).dot(D0_n[n]-Di_2n[n])
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