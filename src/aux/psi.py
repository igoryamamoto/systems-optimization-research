#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:54:45 2017

@author: igor
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
    