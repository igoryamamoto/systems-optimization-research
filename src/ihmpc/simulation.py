# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    
    def __init__(self, controller, tsim):
        self.controller = controller
        self.tsim = tsim
        self.X = np.zeros((tsim + 1, self.controller.nx))
        self.Y = np.zeros((tsim + 1, self.controller.ny+self.controller.nz))
        self.dU = np.zeros((tsim, self.controller.nu))

    def run(self, set_points):
        if set_points.shape[0] != self.tsim:
            raise ValueError("size of set_points doesn't match tsim.")
            
        for k in range(self.tsim):
            self.dU[k] = self.controller.calculate_control(set_points[k])
            self.X[k+1], self.Y[k+1] = self.controller.opom.output(self.dU[k])
            
    def show_results(self, u0=None):
        if not u0:
            u0 = np.zeros(self.controller.nu)
        dU = self.dU
        U = np.concatenate(([u0], dU))
        for k, du in enumerate(dU):
            U[k+1] = U[k] + du
        Y = self.Y
        
        for n in range(self.controller.ny):
            plt.plot(Y[:, n], label='y{}'.format(n+1))
            
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')    
        plt.show()
        
        plt.figure()
        for n in range(self.controller.nu):
            plt.plot(dU[:, n], label='du{}'.format(n+1))
        
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')
        plt.show()
        
        plt.figure()
        for n in range(self.controller.nu):
            plt.plot(U[:, n], label='u{}'.format(n+1))
        
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')
        plt.show()
