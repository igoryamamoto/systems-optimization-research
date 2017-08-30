# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: igoryamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy import signal


class SystemModel(object):
    def __init__(self, H):
        self.H = np.array(H)
        self.ny = self.H.shape[0]
        self.nu = self.H.shape[1]

    def step_response(self, X0=None, T=None, N=None):
        def fun(X02=None, T2=None, N2=None):
            def fun2(sys):
                return signal.step(sys, X02, T2, N2)
            return fun2
        fun3 = fun(X0, T, N)
        step_with_time = list(map(fun3, self.H))
        return [s[1] for s in step_with_time]

class OPOM(SystemModel):
    def __init__(self, H, Ts):
        super().__init__(H)
        self.Ts = Ts
        self.na = 1 # max order of Gij
        self.nd = self.ny*self.nu*self.na
        self.nx = 2*self.ny+self.nd
        self.A, self.B, self.C, self.D, self.D0, self.Di, self.Dd, self.J, self.F, self.N, self.Psi = self._build_OPOM()
    
    def __repr__(self):
        return "A=\n%s\n\nB=\n%s\n\nC=\n%s\n\nD=\n%s" % (self.A.__repr__(), 
                                                         self.B.__repr__(), 
                                                         self.C.__repr__(), 
                                                         self.D.__repr__())
        
    def _get_coeff(self, b, a):
        # multiply by 1/s (step)
        a = np.append(a, 0)
        # do partial fraction expansion
        r,p,k = signal.residue(b,a)
        # r: Residues
        # p: Poles
        # k: Coefficients of the direct polynomial term    
        d_s = np.array([])
        #d_d = np.array([])
        d_d = np.zeros(self.na)
        d_i = np.array([])
        poles = np.zeros(self.na)
        integrador = 0
        i = 0;
        for i in range(np.size(p)):
            if (p[i] == 0):
                if (integrador):
                    d_i = np.append(d_i, r[i])
                else:
                    d_s = np.append(d_s, r[i])
                    integrador += 1
            else:
                d_d[i] = r[i]
                poles[i] = p[i]
                i += 1
        if (d_i.size == 0):
            d_i = np.append(d_i, 0)
        return d_s, d_d, d_i, poles

    def _create_J(self):
        J = np.zeros((self.nu*self.na, self.nu))
        for col in range(self.nu):
            J[col*self.na:col*self.na+self.na, col] = np.ones(self.na)
        return J

    def _create_psi(self):
        Psi = np.zeros((self.ny, self.nd))
        size = self.nu*self.na
        for row in range(self.ny):
            Psi[row, row*size:row*size+size] = np.ones(size)
        return Psi

    def _build_OPOM(self):
        b11 = self.H[0][0].num
        a11 = self.H[0][0].den
        # g12 = -1.7/(19.5*s+1)
        b12 = self.H[0][1].num
        a12 = self.H[0][1].den
        # g21 = -0.763/(31.8*s+1)
        b21 = self.H[1][0].num
        a21 = self.H[1][0].den
        # g22 = 0.235/s
        b22 = self.H[1][1].num
        a22 = self.H[1][1].den
        
        # Obtain coefficients of partial fractions of Gij/s
        d0_11, dd_11, di_11, r_11 = self._get_coeff(b11, a11)
        d0_12, dd_12, di_12, r_12 = self._get_coeff(b12, a12)
        d0_21, dd_21, di_21, r_21 = self._get_coeff(b21, a21)
        d0_22, dd_22, di_22, r_22 = self._get_coeff(b22, a22)
        
        # Define matrices D0[nyxnu]. Dd[ndxnd], Di[nyxnu] for each Gij
        D0 = np.vstack((np.hstack((d0_11, d0_12)),
                       np.hstack((d0_21, d0_22))))
        Di = np.vstack((np.hstack((di_11, di_12)),
                       np.hstack((di_21, di_22))))
        Dd = np.diag(dd_11.tolist() + dd_12.tolist() + dd_21.tolist() + dd_22.tolist());
        
        # Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]
        J = self._create_J()
        
        F = np.diag(np.exp(r_11.tolist() + r_12.tolist() + r_21.tolist() + r_22.tolist()))
        
        N = np.vstack((J, J))
        
        a1 = np.hstack((np.eye(self.ny), np.zeros((self.ny,self.nd)), self.Ts*np.eye(self.ny)))
        a2 = np.hstack((np.zeros((self.nd,self.ny)), F, np.zeros((self.nd,self.ny))))
        a3 = np.hstack((np.zeros((self.ny,self.ny)), np.zeros((self.ny,self.nd)), np.eye(self.ny)))
        A = np.vstack((a1, a2, a3))
        
        B = np.vstack((D0+self.Ts*Di, Dd.dot(F).dot(N), Di))
        
        Psi = self._create_psi()
        C = np.hstack(( np.eye(self.ny), Psi, np.zeros((self.ny,self.ny)) ))
        
        D = np.zeros((self.ny,self.nu))
        return A, B, C, D, D0, Di, Dd, J, F, N, Psi
        
    def output(self, U, T):
        tsim = np.size(T)
        X = np.zeros((tsim,8))
        Y = np.zeros((tsim,2))
        for k in range(tsim-1):
            X[k+1] = self.A.dot(X[k]) + self.B.dot(U[k+1]-U[k])
            Y[k+1] = self.C.dot(X[k])
        plt.plot(Y[:,0], label='y1')
        plt.plot(Y[:,1], label='y2')
        plt.plot(U[1:,0], '--', label='u1')
        plt.plot(U[1:,1], '--', label='u2')
        plt.legend(loc=4)
        plt.savefig('opom_step.png')
        #return U, Y
        
class IHMPCController(OPOM):
    def __init__(self, H, Ts, m):
        # dt, m, ny, nu, na, D0, Dd, Di, F, N, Z, W, Q, R, r, G1, G2, G3
        super().__init__(H, Ts)
        self.system = H
        self.m = 3 # control horizon
                

#TODO: Define matrix Wn[ndxnd.m] from F

#TODO: Define matrix Z[?] from Dd,N

#TODO: Define matrices D0n[nyxm.nu], Di1n[nyxm.nu], D12n[nyxm.nu]

#TODO: Obtain G1, G2, G3

if __name__ == '__main__':
    h11 = signal.TransferFunction([-0.19],[1, 0])
    h12 = signal.TransferFunction([-1.7],[19.5, 1])
    h21 = signal.TransferFunction([-0.763],[31.8, 1])
    h22 = signal.TransferFunction([0.235],[1, 0])
    H = [[h11, h12], [h21, h22]]
    Ts = 1
    m = 3
    controller = IHMPCController(H, Ts, m)
    A = controller.A
    B = controller.B
    C = controller.C
    D = controller.D
    Dd = controller.Dd
    Di = controller.Di
    N = controller.N
    F =controller.F
    tsim = 100
    #U = np.vstack(( [0,0] ,np.ones((tsim-1,2)) ))
    import pickle
    with open('u1.pickle','rb') as f:
        u1 = pickle.load(f)
    with open('u2.pickle','rb') as f:
        u2 = pickle.load(f)
    U = np.vstack((u1,u2)).T
    T = np.arange(tsim)
    
    controller.output(U, T)

##%%
#t_sim = 300
#
## Real Process
#Br11 = np.array([-0.19])
#Ar11 = np.array([1, -1])
#Br12 = np.array([-0.08498])
#Ar12 = np.array([1, -0.95])
#Br21 = np.array([-0.02362])
#Ar21 = np.array([1, -0.969])
#Br22 = np.array([0.235])
#Ar22 = np.array([1, -1])
#
#na11 = len(Ar11)
#na12 = len(Ar12)
#na21 = len(Ar21)
#na22 = len(Ar22)
#nb = 1
#
##%% Set point and Disturbance Signals
#w1 = np.array([1]*(t_sim))
#w2 = np.array([1]*(t_sim))
#
##%% Initialization
#y11 = np.zeros(t_sim+1)
#y12 = np.zeros(t_sim+1)
#y21 = np.zeros(t_sim+1)
#y22 = np.zeros(t_sim+1)
#u1 = np.zeros(t_sim+1)
#u2 = np.zeros(t_sim+1)
#du1 = np.zeros(t_sim+1)
#du2 = np.zeros(t_sim+1)
#y11_past = np.zeros(na11)
#y12_past = np.zeros(na12)
#y21_past = np.zeros(na21)
#y22_past = np.zeros(na22)
#u1_past = np.zeros(nb)
#u2_past = np.zeros(nb)
#
##%% Control Loop
#for k in range(1,t_sim+1):
#    # Real process
#    y11[k] = -Ar11[1:].dot(y11_past[:-1]) + Br11.dot(u1_past)
#    y12[k] = -Ar12[1:].dot(y12_past[:-1]) + Br12.dot(u2_past)
#    y21[k] = -Ar21[1:].dot(y21_past[:-1]) + Br21.dot(u1_past)
#    y22[k] = -Ar22[1:].dot(y22_past[:-1]) + Br22.dot(u2_past)
#
#    # Free Response
#
#
#    # Select set points for the current horizon
#    w = []
#    # Solver Inputs
#    H = matrix()
#    q = matrix()
#    A = matrix()
#    b = matrix()
#    # Solve
#    sol = solvers.qp(P=H,q=q,G=A,h=b)
#    dup = list(sol['x'])
#
#    du1[k] = dup[0]
#    du2[k] = dup[m]
#    u1[k] = u1[k-1] + du1[k]
#    u2[k] = u2[k-1] + du2[k]
#
#    u1_past = np.append(u1[k],u1_past[:-1])
#    u2_past = np.append(u2[k],u2_past[:-1])
#    y11_past = np.append(y11[k],y11_past[:-1])
#    y12_past = np.append(y12[k],y12_past[:-1])
#    y21_past = np.append(y21[k],y21_past[:-1])
#    y22_past = np.append(y22[k],y22_past[:-1])
#
#
##%% Teste
#plt.clf()
#plt.plot([1]*(t_sim+1),':', label='Target')
#plt.plot(y11+y12, label='y1')
#plt.plot(y21+y22, label='y2')
#plt.plot(u1,'--', label='u1')
#plt.plot(u2,'--', label='u2')
#plt.legend(loc=4)
#plt.xlabel('sample time (k)')
