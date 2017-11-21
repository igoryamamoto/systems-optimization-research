# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from cvxopt import matrix, solvers


class SystemModel(object):
    def __init__(self, H):
        self.H = np.array(H)
        self.ny = self.H.shape[0]
        self.nu = self.H.shape[1]

    def output(self, X0=None, T=None, N=None):
        def fun(X02=None, T2=None, N2=None):
            def fun2(sys):
                return signal.step(sys, X02, T2, N2)
            return fun2
        fun3 = fun(X0, T, N)
        step_with_time = list(map(fun3, self.H))
        return np.array([s[1] for s in step_with_time])


class Controller(object):
    def __init__(self, Ts, m, Q, R, du_min, du_max):
        self.Ts = Ts
        self.m = m
        self.Q = Q
        self.R = R
        self.du_min = du_min
        self.du_max = du_max


class Simulation(object):
    def __init__(self, w, controller, real_system=None):
        self.results = {"y": [], "u": []}
        self.w = w
        self.controller = controller
        if real_system:
            self.real_system = real_system
        else:
            self.real_system = controller.system

    def run(self, tsim):
        # Initialization
        y11 = 0*np.ones(tsim+1)
        y12 = 0*np.ones(tsim+1)
        y21 = 0*np.ones(tsim+1)
        y22 = 0*np.ones(tsim+1)
        u1 = np.zeros(tsim+1)
        u2 = np.zeros(tsim+1)
        du1 = np.zeros(tsim+1)
        du2 = np.zeros(tsim+1)
        y11_past = 0*np.ones(na11)
        y12_past = 0*np.ones(na12)
        y21_past = 0*np.ones(na21)
        y22_past = 0*np.ones(na22)
        u1_past = np.zeros(nb)
        u2_past = np.zeros(nb)

        J = np.zeros(tsim)
        Status = ['']*tsim
        # Control Loop
        for k in range(1,tsim+1):
            y11[k] = -Ar11[1:].dot(y11_past[:-1])
                    + Br11.dot(u1_past)
            y12[k] = -Ar12[1:].dot(y12_past[:-1])
                    + Br12.dot(u2_past)
            y21[k] = -Ar21[1:].dot(y21_past[:-1])
                    + Br21.dot(u1_past)
            y22[k] = -Ar22[1:].dot(y22_past[:-1])
                    + Br22.dot(u2_past)

            # Select references for the current horizon
            w = np.append(self.w[0][k:k+p], self.w[1][k:k+p])
            du_past = np.array([du1[k-1], du2[k-1]])
            y_past = [y11_past, y12_past, y21_past, y22_past]
            current_du = [np.array([du1[k]]), np.array([du2[k]])]
            dup,j,s =
            self.controller.calculate_control(w,
                                            du_past=du_past,
                                            y_past=y_past,
                                            current_du=current_du)

            du1[k] = dup[0]
            du2[k] = dup[m]
            u1[k] = u1[k-1] + du1[k]
            u2[k] = u2[k-1] + du2[k]

            u1_past = np.append(u1[k],u1_past[:-1])
            u2_past = np.append(u2[k],u2_past[:-1])
            y11_past = np.append(y11[k],y11_past[:-1])
            y12_past = np.append(y12[k],y12_past[:-1])
            y21_past = np.append(y21[k],y21_past[:-1])
            y22_past = np.append(y22[k],y22_past[:-1])

            J[k-1] = abs(j)
            Status[k-1] = s
        self.results['y'] = [y11+y12, y21+y22]
        self.results['u'] = [u1, u2]

    def show_results():
        plt.clf()
        plt.plot(w1[:-p],':', label='Target y1')
        plt.plot(w2[:-p],':', label='Target y2')
        plt.plot(self.results['y'][0], label='y1')
        plt.plot(self.results['y'][1], label='y2')
        plt.plot(self.results['u'][0],'--', label='u1')
        plt.plot(self.results['u'][0],'--', label='u2')
        plt.legend(loc=0, fontsize='small')
        plt.xlabel('sample time (k)')
        plt.show()
        #plt.savefig('sim8.png')
        return J, Status
