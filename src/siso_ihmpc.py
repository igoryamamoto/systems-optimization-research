# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from ihmpc import OPOM, IHMPCController, Simulation


if __name__ == '__main__':
    # funcao de transferencia 1/(s+1)
    h = signal.TransferFunction([1], [1, 2])
    # funcao de transferencia nula
    h0 = signal.TransferFunction([0], [1])
    # tempo de amostragem
    Ts = 1
    # horizonte de controle
    m = 3
    # construcao do controlador
    controller = IHMPCController([[h, h0], [h0, h0]], Ts, m)
    # numero de amostras da simulacao
    tsim = 100
    # estado inicial
    controller.opom.X = np.array([0,0, 0, 0, 0, 0, 0, 0])
    # construcao objeto de simulacao
    sim = Simulation(controller, tsim)
    # referencia
    ref = np.array([1, 0])
    # chamada da simulacao
    sim.run(ref)
    
    
    # plots
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
    plt.show()