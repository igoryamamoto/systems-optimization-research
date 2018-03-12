# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from ihmpc import OPOM, IHMPCController, Simulation


if __name__ == '__main__':
    h11 = signal.TransferFunction([-0.19], [1, 0])
    h12 = signal.TransferFunction([-1.7], [19.5, 1])
    h21 = signal.TransferFunction([-0.763], [31.8, 1])
    h22 = signal.TransferFunction([0.235], [1, 0])
    H1 = [[h11, h12], [h21, h22]]
    Ts = 1
    m = 3
    controller = IHMPCController(H1, Ts, m)
    tsim = 100
    controller.opom.X = np.array([0,0, 0, 0, 0, 0, 0, 0])
    sim = Simulation(controller, tsim)
    sim.run()
    dU = sim.dU
    Y = sim.Y
    plt.plot(Y[:, 0], label='y1')
    plt.plot(Y[:, 1], label='y2')
    plt.legend(loc=0, fontsize='large')
    plt.xlabel('sample time (k)')    
    plt.show()
    
    plt.figure()
    plt.plot(dU[:, 0], label='du1')
    plt.plot(dU[:, 1], label='du2')
    plt.legend(loc=0, fontsize='large')
    plt.xlabel('sample time (k)')
    
#    A = controller.opom.A
#    B = controller.opom.B
#    C = controller.opom.C
#    D = controller.opom.D
#    D0 = controller.opom.D0
#    Dd = controller.opom.Dd
#    Di = controller.opom.Di
#    N = controller.opom.N
#    F = controller.opom.F
#    Z = controller.Z
#    R = controller.opom.R
#    D0_n = controller.D0_n
#    Di_1n = controller.Di_1n
#    Di_2n = controller.Di_2n
#    Psi = controller.opom.Psi
#    Wn = controller.Wn
#    Aeq = controller.Aeq