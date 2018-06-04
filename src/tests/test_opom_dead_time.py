# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from opom_dead_time import OPOM, TransferFunctionDelay


class OPOMSimulation(object):
    def __init__(self, opom, tsim):
        self.opom = opom
        self.tsim = tsim
        self.X = np.zeros((tsim + 1, self.opom.nx))
        self.Y = np.zeros((tsim + 1, self.opom.ny+self.opom.nz))
        self.dU = np.zeros((tsim, self.opom.nu))

    def step(self):
        self.dU[0] = np.ones((1, self.opom.nu))
        self.X, self.Y = self.opom.output(self.dU, samples=self.tsim)
            
    def show_results(self, u0=None):
        if not u0:
            u0 = np.zeros(self.opom.nu)
        dU = self.dU
        U = np.concatenate(([u0], dU))
        for k, du in enumerate(dU):
            U[k+1] = U[k] + du
        Y = self.Y
        
        for n in range(self.opom.ny):
            plt.plot(Y[:, n], label='y{}'.format(n+1))
            
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')    
        plt.show()
        
        plt.figure()
        for n in range(self.opom.nu):
            plt.plot(dU[:, n], label='du{}'.format(n+1))
        
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')
        plt.show()
        
        plt.figure()
        for n in range(self.opom.nu):
            plt.plot(U[:, n], label='u{}'.format(n+1))
        
        plt.legend(loc=0, fontsize='large')
        plt.xlabel('sample time (k)')
        plt.show()
        
h11 = TransferFunctionDelay([-0.19], [1, 0], delay=20)
h12 = TransferFunctionDelay([-1.7], [19.5, 1], delay=30)
h21 = TransferFunctionDelay([-0.763], [31.8, 1], delay=40)
h22 = TransferFunctionDelay([0.235], [1, 0], delay=50)
H = [[h11, h12], [h21, h22]]

h = TransferFunctionDelay([1], [1, 1], delay=60)
# h = TransferFunctionDelay([-1.7], [19.5, 1], delay=20)
H = [[h, h, h], [h, h, h], [h, h, h]]
Ts = 1
model = OPOM(H, Ts)

tsim = 150
sim = OPOMSimulation(model, tsim)
sim.step()
sim.show_results()
