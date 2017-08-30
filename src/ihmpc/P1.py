#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:05:41 2017

@author: Ígor Yamamoto
"""

# m: control horizon
# n: iteration variable
import numpy as np
from scipy.linalg import block_diag
#from numpy.linalg import inv
# revisar dimensao

z = Dd.dot(N)
Z = z
for i in range(m-1):
    Z = block_diag(Z,z)

Q = np.eye(m)


def faz_D0n_ou_Di1n(D, n, m):
    D_n = D
    for i in range(m-1):
        if i >= n-1:
            d_n = np.zeros(D.shape)
        else:
            d_n = D
        D_n = np.concatenate((D_n,d_n), axis=1)
    return D_n

def faz_Di2n(D, n, m):
    D_n = np.zeros(D.shape)
    for i in range(1, m):
        if i > n:
            d_n = np.zeros(D.shape)
        else:
            d_n = i*Ts*D
        D_n = np.concatenate((D_n,d_n), axis=1)
    return D_n

# F é sempre diagonal!
def faz_Wn(F, n, m):
    Wn = np.eye(F.shape[0])
    for i in range(1, m):
        if i > n-1:
            wn = np.zeros(F.shape)
        else:
            f = np.diag(F)
            wn = np.diag(f ** (-i))
            
        Wn = np.concatenate((Wn,wn), axis=1)
    return Wn

D0_n = []
for i in range(1,m+1):
    D0_n.append(faz_D0n_ou_Di1n(D0, i, m))

Di_1n = []
for i in range(1,m+1):
    Di_1n.append(faz_D0n_ou_Di1n(Di, i, m))

Di_2n = []
for i in range(m):
    Di_2n.append(faz_Di2n(Di, i, m))

Wn = []
for i in range(1, m+1):
    Wn.append(faz_Wn(F, i, m))

H_m = 0
for n in range(m):
    a = Z.T.dot(W_n.T).dot(G2(n)-G2(n-1)).dot(W_n).dot(Z)
    b1 = (G1(n)-G1(n-1)).T.dot(Q).dot(D0_n-Di_2n)
    b2 = (G3(n)-G3(n-1)).T.dot(Q).dot(Di_1n)
    b3 = (G1(n)-G1(n-1)).T.dot(Q).dot(Di_1n)
    b = 2*Z.T.dot(W_n.T).dot(b1+b2+b3)
    c1 = Ts*(D0_n-Di_2n).T.dot(Q).dot(D0_n-Di_2n)
    c2 = 2*(n-1/2)*Ts**2*(D0_n-Di_2n).T.dot(Q).dot(Di_1n)
    c3 = (n**3-n+1/3)*Ts**3*Di_1n.T.dot(Q).dot(Di_1n)
    c = c1 + c2 + c3
    H_m += a + b + c

H_inf = Z.T.dot(W_m.T).dot(G2_inf-G2_m).dot(W_m).dot(Z)

H = H_m + H_inf

cf_m = 0
for n in range(m):
    a = (-e_s.T.dot(Q).dot(G1(n)-G1(n-1)) + x_d.T * (G2(n)-G2(n-1)) + x_i.T.dot(Q.dot(G3(n)-G3(n-1))))*W_n*Z
    b = (-Ts*e_s.T + (n-1/2)*Ts**2*x_i.T+x_d.T*(G1(n)-G1(n-1)).T).dot(Q).dot(D0_n-Di_2n)
    c = (-(n-1/2)*Ts**2*e_s.T + (n**3-n+1/3)*Ts**3*x_i.T + x_d.T*(G3(n)-G3(n-1)).T).dot(Q).dot(Di_1n)
    cf_m += a + b + c 

cf_inf = x_d.T*(G2_inf-G2_m).dot(W_m).dot(Z)

cf = cf_m + cf_inf


'''
Questionamentos:
     - operador xzinho
     - dimensao de cf -> verificar dimensao de cf_inf -> dimensao de d_u, que tem dimensao m
     - inputs: estado atual de x_s, x_d, x_i, e_s, referencia
     - G1, G2 e G3 sao funcoes? que retornam matrizes -> fazer um vetor?
     - ficar atento ao sample time
     - 
'''