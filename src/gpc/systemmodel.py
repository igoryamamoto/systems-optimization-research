# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import scipy.signal as signal


class SystemModel(object):
    def __init__(self, ny, nu, H):
        self.ny = ny
        self.nu = nu
        self.H = np.array(H)

    def step_response(self, X0=None, T=None, N=None):
        def fun(X02=None, T2=None, N2=None):
            def fun2(sys):
                return signal.step(sys, X02, T2, N2)
            return fun2
        fun3 = fun(X0, T, N)
        step_with_time = list(map(fun3, self.H))
        return np.array([s[1] for s in step_with_time])