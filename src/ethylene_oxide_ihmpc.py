# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np

from ihmpc import IHMPCController, Simulation


if __name__ == '__main__':
    h11 = signal.TransferFunction([-0.19], [1, 0])
    h12 = signal.TransferFunction([-1.7], [19.5, 1])
    h21 = signal.TransferFunction([-0.763], [31.8, 1])
    h22 = signal.TransferFunction([0.235], [1, 0])
    H1 = [[h11, h12], [h21, h22]]
    Ts = 1
    m = 3
    du_max = 0.7
    Q = np.array([[1, 0],
                  [0, 1]])
    R = np.array([[1e-2, 0],
                  [0, 1e-2]])
    
    controller = IHMPCController(H1, Ts, m, du_max, Q, R)
    controller.opom.X = np.array([0,0, 0, 0, 0, 0, 0.4, -0.4])
    
    
    tsim = 200
    step_time = int(tsim/2)
    set_points = np.array([[0, 0]]*step_time + [[2, 2]]*(tsim - step_time))
    
    sim = Simulation(controller, tsim)
    sim.run(set_points)
    sim.show_results()
