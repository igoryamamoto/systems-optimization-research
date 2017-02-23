# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:58:03 2017

@author: Ígor Yamamoto
"""
import numpy as np

def create_matrix_G(g, P, Nu):
    g = np.append(g,np.zeros(Nu-1))
    G = np.array([])
    for i in range(P):
        G = np.append(G,[g[i-j] for j in range(Nu)])  
    return np.resize(G,(P,Nu))