# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto
"""
import numpy as np


class Simulation(object):
    
    def __init__(self, controller, tsim):
        self.controller = controller
        self.tsim = tsim
        self.X = np.zeros((tsim + 1, self.controller.nx))
        self.Y = np.zeros((tsim + 1, self.controller.ny))
        self.dU = np.zeros((tsim, self.controller.nu))

    def run(self, ref):
        for k in range(self.tsim):
            self.dU[k] = self.controller.calculate_control(ref)
            self.X[k+1], self.Y[k+1] = self.controller.opom.output(self.dU[k])
