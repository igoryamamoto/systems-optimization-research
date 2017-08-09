# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 08:23:31 2016

Este script realiza a conversão do modelo de função de transferência para o modelo OPOM (Output Prediction Oriented Model)

@author: bruno eduardo benetti (brunobillbenetti@gmail.com)
"""

#Módulos utilizados
import numpy as np
from scipy import signal


#define ft
b = np.array([1])  #numerador
a = np.array([1, 60, 500]) #denominador

#define tempo de amostragem
Ts = 1

#multiplica por 1/s
sizeb = np.size(a)
a = np.insert(a, sizeb, 0)

#faz frações parciais
r,p,k = signal.residue(b,a)
# r = coeficientes
# p = polos
# k = Coefficients of the direct polynomial term.?????

#extrai coeficiente de steady state
#extrai coeficiente de dinâmica
#extrai polos
d_s = np.array([])
d_d = np.array([])
d_i = np.array([])
polos = np.array([])
integrador = 0

for i in range(np.size(p)):
    if (p[i]==0):
        if (integrador):
            d_i = np.insert(d_i,np.size(d_i), r[i])
        else:
            d_s = np.insert(d_s, np.size(d_s), r[i])
            integrador += 1
    else:
        d_d = np.insert(d_d, np.size(d_d), r[i])
        polos = np.insert(polos, np.size(polos), p[i])

if (np.size(d_i)==0):
    d_i = np.insert(d_i,np.size(d_i), 0)


#monta matriz A
A_1 = np.hstack((np.array([[1]]),np.zeros((1,2)),np.array([[Ts]])))  #primeira linha
z_1 = np.zeros((np.size(d_d),np.size(d_s)))
z_2 = np.zeros((np.size(d_d),np.size(d_i)))
A_2 = np.hstack((z_1,np.diag(np.exp(Ts/polos),0),z_2)) #segunda linha
A_3 = np.hstack((np.zeros((1,np.size(d_s))), np.zeros((1,np.size(d_d))), np.array([[1]]))) #terceira linha


A = np.vstack((A_1,A_2,A_3))


#monta matriz B

B_1 = d_s+d_i
B_2 = np.reshape(d_d*np.exp(Ts/polos),(np.size(d_d),1))
B_3 = d_i

B = np.vstack((B_1,B_2,B_3))
#monta matriz C
C = np.vstack((np.ones((np.size(d_d)+1,1)),np.array(([0]))))

D = np.array([[0]])

#monta sistema ss
sys = signal.StateSpace(A, B, C, D)


