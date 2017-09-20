# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy import signal
from scipy.linalg import block_diag


class OPOM(object):
    def __init__(self, H, Ts):
        self.H = np.array(H)
        self.ny = self.H.shape[0]
        self.nu = self.H.shape[1]
        self.Ts = Ts
        self.na = 1 # max order of Gij
        self.nd = self.ny*self.nu*self.na
        self.nx = 2*self.ny+self.nd
        self.X = np.zeros(self.nx)
        self.A, self.B, self.C, self.D, self.D0, self.Di, self.Dd, self.J, self.F, self.N, self.Psi, self.R = self._build_OPOM()
    
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
        
        R = np.array([r_11, r_12, r_21, r_22])
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
        
        def psi(t):
            R2 = np.array(list(map(lambda x: np.exp(x*t), R)))
            psi = np.zeros((self.ny, self.nd))
            for i in range(self.ny):
                phi = np.array([])
                for j in range(self.nu):
                    phi = np.concatenate((phi, R2[i*self.nu+j]))
                psi[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi      
            return psi
        
        C = lambda t: np.hstack(( np.eye(self.ny), psi(t), np.eye(self.ny)*t))
        
        D = np.zeros((self.ny,self.nu))
        return A, B, C, D, D0, Di, Dd, J, F, N, psi, R
        
    def output(self, U, T):
        tsim = np.size(T)
        X = np.zeros((tsim,8))
        Y = np.zeros((tsim,2))
        for k in range(tsim-1):
            X[k+1] = self.A.dot(X[k]) + self.B.dot(U[k+1]-U[k])
            Y[k+1] = self.C(0).dot(X[k])
        plt.plot(Y[:,0], label='y1')
        plt.plot(Y[:,1], label='y2')
        plt.plot(U[1:,0], '--', label='u1')
        plt.plot(U[1:,1], '--', label='u2')
        plt.legend(loc=4)
        plt.savefig('../../img/opom_step.png')
        plt.show()
        #return U, Y
        
class IHMPCController(OPOM):
    def __init__(self, H, Ts, m):
        # dt, m, ny, nu, na, D0, Dd, Di, F, N, Z, W, Q, R, r, G1, G2, G3
        super().__init__(H, Ts)
        self.m = m # control horizon
        self.Q = np.eye(self.ny)
        self.Z, self.D0_n, self.Di_1n, self.Di_2n, self.Wn = self._make_matrices()
    
    def _make_matrices(self):
        def faz_D0n_ou_Di1n(D, n, m):
            D_n = D
            for i in range(m-1):
                if i >= n-1:
                    d_n = np.zeros(D.shape)
                else:
                    d_n = D
                D_n = np.concatenate((D_n,d_n), axis=1)
            return D_n
    
        def faz_Di2n(D, n, m):
            D_n = np.zeros(D.shape)
            for i in range(1, m):
                if i > n:
                    d_n = np.zeros(D.shape)
                else:
                    d_n = i*Ts*D
                D_n = np.concatenate((D_n,d_n), axis=1)
            return D_n
        
        def faz_Wn(F, n, m):
            Wn = np.eye(F.shape[0])
            for i in range(1, m):
                if i > n-1:
                    wn = np.zeros(F.shape)
                else:
                    f = np.diag(F)
                    wn = np.diag(f ** (-i))
                    
                Wn = np.concatenate((Wn,wn), axis=1)
            return Wn
    
        z = self.Dd.dot(self.N)
        Z = z
        for i in range(self.m - 1):
            Z = block_diag(Z,z)
            
        D0_n = []
        for i in range(1, self.m + 1):
            D0_n.append(faz_D0n_ou_Di1n(self.D0, i, self.m))
        
        Di_1n = []
        for i in range(1, self.m + 1):
            Di_1n.append(faz_D0n_ou_Di1n(self.Di, i, self.m))
        
        Di_2n = []
        for i in range(m):
            Di_2n.append(faz_Di2n(self.Di, i, self.m))
        
        Wn = []
        for i in range(1, self.m + 1):
            Wn.append(faz_Wn(self.F, i, self.m))
            
        return Z, D0_n, Di_1n, Di_2n, Wn
    
    
    def control(self):
        def G1(self, n):
            G = np.zeros((self.ny, self.nd))
            for i in range(self.ny):
                phi = np.array([])
                for j in range(self.nu):
                    r = self.R[i*self.nu+j]
                    if r == 0:
                        g = n
                    else:
                        g = 1/r*(np.exp(r*n)-1)
                    phi = np.append(phi, g)
                G[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi      
            return G
    
        def G2(self, n):
            G = np.array([])
            for y in range(self.ny):
                r = self.R[y*self.nu:y*self.nu+self.nu]
                g = np.zeros((self.nu, self.nu))
                for i in range(self.nu):
                    for j in range(self.nu):
                        g[i, j] = r[i] + r[j]
                G = block_diag(G, g)
            return G[1:]
        
        def G3(self, n):
            G = np.zeros((self.ny, self.nd))
            for i in range(self.ny):
                phi = np.array([])
                for j in range(self.nu):
                    r = self.R[i*self.nu+j]
                    phi = np.append(phi, r)
                if 0 in phi:
                    def aux(x):
                        if x==0:
                            return 0
                        else:
                            return (1/x**2)*np.exp(x*n)*(x*n-1)
                    phi = np.array(list(map(aux, phi)))
                else:
                    phi = np.zeros(nu)
                G[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi 
            return G
            
            
            
            
        H_m = 0
        for n in range(m):
            a = self.Z.T.dot(self.Wn[n].T).dot(G2(n) - G2(n-1)).dot(self.Wn[n]).dot(self.Z)
            b1 = (G1(n) - G1(n-1)).T.dot(self.Q).dot(self.D0_n[n] - self.Di_2n[n])
            b2 = (G3(n) - G3(n-1)).T.dot(self.Q).dot(self.Di_1n[n])
            b3 = (G1(n) - G1(n-1)).T.dot(self.Q).dot(self.Di_1n[n])
            b = 2*self.Z.T.dot(self.Wn[n].T).dot(b1 + b2 + b3)
            c1 = self.Ts[n]*(self.D0_n[n] - self.Di_2n[n]).T.dot(self.Q).dot(self.D0_n[n] - self.Di_2n[n])
            c2 = 2*(n - 1/2)*Ts**2*(self.D0_n[n] - self.Di_2n[n]).T.dot(self.Q).dot(self.Di_1n[n])
            c3 = (n**3 - n + 1/3)*Ts**3*self.Di_1n[n].T.dot(self.Q).dot(self.Di_1n[n])
            c = c1 + c2 + c3
            H_m += a + b + c
        H_inf = self.Z.T.dot(self.W_m.T).dot(G2_inf - G2_m).dot(self.W_m).dot(self.Z)
        H = H_m + H_inf
        
        cf_m = 0
        for n in range(m):
            a = (-e_s.T.dot(self.Q).dot(G1(n)-G1(n-1)) + x_d.T * (G2(n)-G2(n-1)) + x_i.T.dot(self.Q.dot(G3(n)-G3(n-1))))*self.Wn*self.Z
            b = (-self.Ts*e_s.T + (n - 1/2)*self.Ts**2*x_i.T + x_d.T*(G1(n)-G1(n-1)).T).dot(self.Q).dot(self.D0_n - self.Di_2n)
            c = (-(n - 1/2)*self.Ts**2*e_s.T + (n**3 - n + 1/3)*self.Ts**3*x_i.T + x_d.T*(G3(n) - G3(n - 1)).T).dot(self.Q).dot(self.Di_1n[n])
            cf_m += a + b + c 
        cf_inf = x_d.T*(G2_inf - G2_m).dot(W_m).dot(self.Z)
        cf = cf_m + cf_inf
        
        sol = solvers.qp(P=H, q=cf)
        # minimize    (1/2)*x'*P*x + q'*x 
        # subject to  G*x <= h      
        #             A*x = b.
        du = list(sol['x'])
        # s = sol['status']
        # j = sol['primal objective']
        return du

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
    D0 = controller.D0
    Dd = controller.Dd
    Di = controller.Di
    N = controller.N
    F = controller.F
    Z = controller.Z
    R = controller.R
    D0_n = controller.D0_n
    Di_1n = controller.Di_1n
    Di_2n = controller.Di_2n
    Psi = controller.Psi
    Wn = controller.Wn
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
    