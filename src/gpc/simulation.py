# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    def __init__(self, controller, real_system=None):
        self.controller = controller
        if real_system:
            self.real_system = real_system
        else:
            self.real_system = controller.system

    def run(self, tsim):
        # Real Process
        Br11 = 1*np.array([-0.19])
        Ar11 = np.array([1, -1])
        Br12 = 1*np.array([-0.08498])
        Ar12 = np.array([1, -0.95])
        Br21 = 1*np.array([-0.02362])
        Ar21 = np.array([1, -0.969])
        Br22 = 1*np.array([0.235])
        Ar22 = np.array([1, -1])
        na11 = len(Ar11)
        na12 = len(Ar12)
        na21 = len(Ar21)
        na22 = len(Ar22)
        nb = 1
        # Reference and Disturbance Signals
        w1 = np.array([1]*int(tsim/4)+[1]*int(tsim/4)+[1]*int(tsim/4)+[1]*int(tsim/4+self.controller.p))
        w2 = np.array([1]*int(tsim/4)+[1]*int(tsim/4)+[1]*int(tsim/4)+[1]*int(tsim/4+self.controller.p))
        #a1 = list(0.0125*np.arange(int(tsim/2)))
        #b1 = [1.25]*int(tsim/4)
        #c1 = list(b1-0.025*np.arange(int(tsim/4)))
        #c1.reverse()
        #d1 = [1.25]*int(tsim/2+p)
        #w1 = np.array(a1+d1)
        
        #a1 = list(0.01*np.arange(int(tsim/2)))
        #b1 = [1]*int(tsim/4)
        #c1 = list(b1-0.02*np.arange(int(tsim/4)))
        #c1.reverse()
        #d1 = [1]*int(tsim/2+p)
        #w2 = np.array(a1+d1)
        
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
            y11[k] = -Ar11[1:].dot(y11_past[:-1]) + Br11.dot(u1_past)
            y12[k] = -Ar12[1:].dot(y12_past[:-1]) + Br12.dot(u2_past)
            y21[k] = -Ar21[1:].dot(y21_past[:-1]) + Br21.dot(u1_past)
            y22[k] = -Ar22[1:].dot(y22_past[:-1]) + Br22.dot(u2_past)

            # Select references for the current horizon
            w = np.append(w1[k:k+self.controller.p], w2[k:k+self.controller.p])

            du_past = np.array([du1[k-1], du2[k-1]])
            y_past = [y11_past, y12_past, y21_past, y22_past]
            current_du = [np.array([du1[k]]), np.array([du2[k]])]
            dup,j,s = self.controller.calculate_control(w,
                                                    du_past=du_past,
                                                    y_past=y_past,
                                                    current_du=current_du)

            du1[k] = dup[0]
            du2[k] = dup[self.controller.m]
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
        # Teste
        plt.clf()
        plt.plot(w1[:-self.controller.p],':', label='Target y1')
        plt.plot(w2[:-self.controller.p],':', label='Target y2')
        plt.plot(y11+y12, label='y1')
        plt.plot(y21+y22, label='y2')
        plt.plot(u1,'--', label='u1')
        plt.plot(u2,'--', label='u2')
        plt.legend(loc=0, fontsize='small')
        plt.xlabel('sample time (k)')    
        plt.show()
        #plt.savefig('sim8.png')
        return J, Status
