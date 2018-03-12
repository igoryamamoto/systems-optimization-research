# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:03 2017

@author: Ígor Yamamoto
"""
from gpc import SystemModel, GPCController, Simulation
import scipy.signal as signal
from cvxopt import solvers
import numpy as np

nu = 2    # number of inputs
ny = 2    # number of outputs
h11 = signal.TransferFunction([-0.19],[1, 0])
h12 = signal.TransferFunction([-1.7],[19.5, 1])
h21 = signal.TransferFunction([-0.763],[31.8, 1])
h22 = signal.TransferFunction([0.235],[1, 0])
ethylene = SystemModel(2, 2, [h11, h12, h21, h22])

p = 10    # prediction horizon
m = 3   # control horizon
Q = 1*np.eye(p*ny)
R = 10*np.eye(m*nu)
du_max = 0.2
du_min = -0.2
Ts = 1
controller = GPCController(ethylene, Ts, p, m, Q, R, du_min, du_max)

real_h11 = signal.TransferFunction([-0.19],[1, 0])
real_h12 = signal.TransferFunction([-1.7],[19.5, 1])
real_h21 = signal.TransferFunction([-0.763],[31.8, 1])
real_h22 = signal.TransferFunction([0.235],[1, 0])
real_ethylene = SystemModel(2, 2, [real_h11, real_h12, real_h21, real_h22])

solvers.options['show_progress'] = False
tsim = 100
sim = Simulation(controller)
J, S = sim.run(tsim)
#plt.plot(J)