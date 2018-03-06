# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto
"""
import numpy as np
import matplotlib.pyplot as plt
# from cvxopt import matrix, solvers
from scipy import signal
from scipy.linalg import block_diag
import scipy.sparse as sparse
import osqp


class OPOM(object):
    def __init__(self, H, Ts):
        self.H = np.array(H)
        self.ny = self.H.shape[0]
        self.nu = self.H.shape[1]
        self.Ts = Ts
        self.na = self._max_order()  # max order of Gij
        self.nd = self.ny*self.nu*self.na
        self.nx = 2*self.ny+self.nd
        self.X = np.zeros(self.nx)
        self.R, self.D0, self.Di, self.Dd, self.F, self.N = self._create_matrices()
        self.A, self.B, self.C, self.D = self._create_state_space()

    def __repr__(self):
        return "A=\n%s\n\nB=\n%s\n\nC=\n%s\n\nD=\n%s" % (self.A.__repr__(),
                                                         self.B.__repr__(),
                                                         self.C.__repr__(),
                                                         self.D.__repr__())

    def _max_order(self):
        na = 0
        for h in self.H.flatten():
            na_h = len(h.den) - 1
            na = max(na, na_h)
        return na

    def _get_coeff(self, b, a):
        # multiply by 1/s (step)
        a = np.append(a, 0)
        # do partial fraction expansion
        r, p, k = signal.residue(b, a)
        # r: Residues
        # p: Poles
        # k: Coefficients of the direct polynomial term
        d_s = np.array([])
        # d_d = np.array([])
        d_d = np.zeros(self.na)
        d_i = np.array([])
        poles = np.zeros(self.na)
        integrador = 0
        i = 0
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

    def _create_matrices(self):
        D0 = np.zeros((self.nu))
        Dd = np.array([])
        Di = np.zeros((self.nu))
        R = np.array([])
        for i in range(self.ny):
            d0_x = np.array([])
            di_x = np.array([])
            for j in range(self.nu):
                b = self.H[i][j].num
                a = self.H[i][j].den
                d0, dd, di, r = self._get_coeff(b, a)
                d0_x = np.hstack((d0_x, d0))
                Dd = np.append(Dd, dd)
                di_x = np.hstack((di_x, di))
                R = np.append(R, r)
            D0 = np.vstack((D0, d0_x))
            Di = np.vstack((Di, di_x))
        Dd = np.diag(Dd)
        D0 = D0[1:]
        Di = Di[1:]

        # Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]
        F = np.diag(np.exp(R))

        J = np.zeros((self.nu*self.na, self.nu))
        for col in range(self.nu):
            J[col*self.na:col*self.na+self.na, col] = np.ones(self.na)

        N = J
        for _ in range(self.ny-1):
            N = np.vstack((N, J))

        return R, D0, Di, Dd, F, N

    def Psi(self, t):
        R2 = np.array(list(map(lambda x: np.exp(x*t), self.R)))
        psi = np.zeros((self.ny, self.nd))
        for i in range(self.ny):
            phi = np.array([])
            for j in range(self.nu):
                phi = np.append(phi, R2[(i*self.nu+j)*self.na:
                                        (i*self.nu + j + 1)*self.na])
            psi[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi
        return psi

    def _create_state_space(self):
        a1 = np.hstack((np.eye(self.ny),
                        np.zeros((self.ny, self.nd)),
                        self.Ts*np.eye(self.ny)))
        a2 = np.hstack((np.zeros((self.nd, self.ny)),
                        self.F,
                        np.zeros((self.nd, self.ny))))
        a3 = np.hstack((np.zeros((self.ny, self.ny)),
                        np.zeros((self.ny, self.nd)),
                        np.eye(self.ny)))
        A = np.vstack((a1, a2, a3))

        B = np.vstack((self.D0+self.Ts*self.Di,
                       self.Dd.dot(self.F).dot(self.N),
                       self.Di))

        def C(t):
            return np.hstack((np.eye(self.ny), self.Psi(t), np.eye(self.ny)*t))

        D = np.zeros((self.ny, self.nu))

        return A, B, C, D

    def output(self, dU, samples=1):
        try:
            shape = dU.shape[1]
            print(shape)
        except IndexError:
            dU = np.reshape(dU,(1,self.nu))
        X = np.zeros((samples+1, self.nx))
        X[0] = self.X
        Y = np.zeros((samples+1, self.ny))
        Y[0] = self.C(0).dot(X[0])
        for k in range(samples):
            X[k+1] = self.A.dot(X[k]) + self.B.dot(dU[k])
            Y[k+1] = self.C(0).dot(X[k+1])

        self.X = X[samples]
        return X[samples], Y[samples]


class IHMPCController(object):
    def __init__(self, H, Ts, m):
        # dt, m, ny, nu, na, D0, Dd, Di, F, N, Z, W, Q, R, r, G1, G2, G3
        self.Ts = Ts
        self.opom = OPOM(H, Ts)
        self.ny = self.opom.ny
        self.nu = self.opom.nu
        self.na = self.opom.na
        self.nd = self.opom.nd
        self.nx = self.opom.nx
        self.m = m  # control horizon
        self.Q = np.eye(self.ny)
        self.Z, self.D0_n, self.Di_1n, self.Di_2n, self.Wn, self.Aeq = self._create_matrices()

    def _create_matrices(self):
        def faz_D0n_ou_Di1n(D, n, m):
            D_n = D
            for i in range(m-1):
                if i >= n-1:
                    d_n = np.zeros(D.shape)
                else:
                    d_n = D
                D_n = np.concatenate((D_n, d_n), axis=1)
            return D_n

        def faz_Di2n(D, n, m):
            D_n = np.zeros(D.shape)
            for i in range(1, m):
                if i > n:
                    d_n = np.zeros(D.shape)
                else:
                    d_n = i*Ts*D
                D_n = np.concatenate((D_n, d_n), axis=1)
            return D_n

        def faz_Wn(F, n, m):
            Wn = np.eye(F.shape[0])
            for i in range(1, m):
                if i > n-1:
                    wn = np.zeros(F.shape)
                else:
                    f = np.diag(F)
                    wn = np.diag(f ** (-i))

                Wn = np.concatenate((Wn, wn), axis=1)
            return Wn

        z = self.opom.Dd.dot(self.opom.N)
        Z = z
        for i in range(self.m - 1):
            Z = block_diag(Z, z)

        D0_n = []
        for i in range(1, self.m + 1):
            D0_n.append(faz_D0n_ou_Di1n(self.opom.D0, i, self.m))

        Di_1n = []
        for i in range(1, self.m + 1):
            Di_1n.append(faz_D0n_ou_Di1n(self.opom.Di, i, self.m))

        Di_2n = []
        for i in range(m):
            Di_2n.append(faz_Di2n(self.opom.Di, i, self.m))

        Wn = []
        for i in range(1, self.m + 1):
            Wn.append(faz_Wn(self.opom.F, i, self.m))

        Di_1m = self.opom.Di
        for _ in range(m-1):
            Di_1m = np.hstack((Di_1m, self.opom.Di))

        Di_3m = self.m*self.Ts*Di_1m-Di_2n[self.m-1]
        D0_m = D0_n[self.m-1]
        Di_1m = Di_1n[self.m-1]

        Aeq = np.vstack((D0_m+Di_3m, Di_1m))

        return Z, D0_n, Di_1n, Di_2n, Wn, Aeq

    def calculate_control(self, ref, X=None):
        
        if X == None:
            X = self.opom.X
        x_s = X[:2]
        x_d = X[2:6]
        x_i = X[6:]
        
        e_s = ref - x_s
        
        def G1(n):
            G = np.zeros((self.ny, self.nd))
            for i in range(self.ny):
                phi = np.array([])
                for j in range(self.nu):
                    r = self.opom.R[i*self.nu+j]
                    if r == 0:
                        g = n
                    else:
                        g = 1/r*(np.exp(r*n)-1)
                    phi = np.append(phi, g)
                G[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi
            return G

        def G2(n):
            G = np.array([])
            for y in range(self.ny):
                r = self.opom.R[y*self.nu:y*self.nu+self.nu]
                g = np.zeros((self.nu, self.nu))
                for i in range(self.nu):
                    for j in range(self.nu):
                        gzin = r[i] + r[j]
                        if gzin == 0:
                            g[i, j] = 0
                        else:
                            g[i, j] = 1/gzin*(np.exp(gzin*n)-1)
                G = block_diag(G, g)
            return G[1:]

        def G3(n):
            G = np.zeros((self.ny, self.nd))
            for i in range(self.ny):
                phi = np.array([])
                for j in range(self.nu):
                    r = self.opom.R[i*self.nu+j]
                    phi = np.append(phi, r)
                if 0 in phi:
                    def aux(x):
                        if x == 0:
                            return n
                        else:
                            return (1/x**2)*np.exp(x*n)*(x*n-1)
                    phi = np.array(list(map(aux, phi)))
                else:
                    phi = np.zeros(self.nu)
                G[i, i*self.nu*self.na:(i+1)*self.nu*self.na] = phi
            return G

        H_m = 0
        for n in range(self.m):
            a = self.Z.T.dot(self.Wn[n].T).dot(G2(n)-G2(n-1)).dot(self.Wn[n]).dot(self.Z)
            b1 = (G1(n) - G1(n-1)).T.dot(self.Q).dot(self.D0_n[n]-self.Di_2n[n])
            b2 = (G3(n)-G3(n-1)).T.dot(self.Q).dot(self.Di_1n[n])
            b3 = (G1(n)-G1(n-1)).T.dot(self.Q).dot(self.Di_1n[n])
            b = 2*self.Z.T.dot(self.Wn[n].T).dot(b1+b2+b3)
            c1 = Ts*(self.D0_n[n]-self.Di_2n[n]).T.dot(self.Q).dot(self.D0_n[n]-self.Di_2n[n])
            c2 = 2*(n-1/2)*Ts**2*(self.D0_n[n]-self.Di_2n[n]).T.dot(self.Q).dot(self.Di_1n[n])
            c3 = (n**3-n+1/3)*Ts**3*self.Di_1n[n].T.dot(self.Q).dot(self.Di_1n[n])
            c = c1 + c2 + c3
            H_m += a + b + c

        H_inf = self.Z.T.dot(self.Wn[m-1].T).dot(G2(float('inf'))-G2(m)).dot(self.Wn[m-1]).dot(self.Z)

        H = H_m + H_inf

        cf_m = 0
        for n in range(self.m):
            a = (-e_s.T.dot(self.Q).dot(G1(n)-G1(n-1)) + x_d.T.dot(G2(n)-G2(n-1)) + x_i.T.dot(self.Q).dot(G3(n)-G3(n-1))).dot(self.Wn[n]).dot(self.Z)
            b = (-Ts*e_s.T + (n-1/2)*Ts**2*x_i.T + x_d.T.dot((G1(n) - G1(n-1)).T)).dot(self.Q).dot(self.D0_n[n]-self.Di_2n[n])
            c = (-(n-1/2)*Ts**2*e_s.T + (n**3-n+1/3)*Ts**3*x_i.T + x_d.T.dot((G3(n)-G3(n-1)).T)).dot(self.Q).dot(self.Di_1n[n])
            cf_m += a + b + c

        cf_inf = x_d.T.dot(G2(float('inf'))-G2(self.m)).dot(self.Wn[m-1]).dot(self.Z)

        cf = cf_m + cf_inf
        beq = np.hstack((e_s - self.m*self.Ts*x_i, -x_i)).T
        # sol = solvers.qp(P=matrix(H), q=matrix(cf), A=matrix(self.Aeq), b=matrix(beq))
        # minimize    (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #             A*x = b.
        # du = list(sol['x'])
        # s = sol['status']
        # j = sol['primal objective']
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(H), q=cf, A=sparse.csc_matrix(self.Aeq), u=beq, verbose=False)
        results = solver.solve()
        du1 = results.x[:3]
        du2 = results.x[3:]
        dU = np.array([du1[0], du2[0]])
        return dU


class Simulation(object):
    
    def __init__(self, controller, tsim):
        self.controller = controller
        self.tsim = tsim
        self.X = np.zeros((tsim + 1, self.controller.nx))
        self.Y = np.zeros((tsim + 1, self.controller.ny))
        self.dU = np.zeros((tsim, self.controller.nu))

    def run(self):
        ref = np.array([1, 1])
        for k in range(self.tsim):
            self.dU[k] = self.controller.calculate_control(ref)
            self.X[k+1], self.Y[k+1] = controller.opom.output(self.dU[k])
        

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
