# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto
"""
import numpy as np
# from cvxopt import matrix, solvers
from scipy.linalg import block_diag
import scipy.sparse as sparse
import osqp
from ihmpc.opom import OPOM


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
                    d_n = i*self.Ts*D
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
        for i in range(self.m):
            Di_2n.append(faz_Di2n(self.opom.Di, i, self.m))

        Wn = []
        for i in range(1, self.m + 1):
            Wn.append(faz_Wn(self.opom.F, i, self.m))

        Di_1m = self.opom.Di
        for _ in range(self.m-1):
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
                        g = 1/r*(1-np.exp(r*n*self.Ts))
                        # g = 1/r*(np.exp(r*n)-1)
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
            c1 = self.Ts*(self.D0_n[n]-self.Di_2n[n]).T.dot(self.Q).dot(self.D0_n[n]-self.Di_2n[n])
            c2 = 2*(n-1/2)*self.Ts**2*(self.D0_n[n]-self.Di_2n[n]).T.dot(self.Q).dot(self.Di_1n[n])
            c3 = (n**3-n+1/3)*self.Ts**3*self.Di_1n[n].T.dot(self.Q).dot(self.Di_1n[n])
            c = c1 + c2 + c3
            H_m += a + b + c

        H_inf = self.Z.T.dot(self.Wn[self.m-1].T).dot(G2(float('inf'))-G2(self.m)).dot(self.Wn[self.m-1]).dot(self.Z)
        
        H = H_m + H_inf
        #H = H_m
        
        cf_m = 0
        for n in range(self.m):
            a = (-e_s.T.dot(self.Q).dot(G1(n)-G1(n-1)) + x_d.T.dot(G2(n)-G2(n-1)) + x_i.T.dot(self.Q).dot(G3(n)-G3(n-1))).dot(self.Wn[n]).dot(self.Z)
            b = (-self.Ts*e_s.T + (n-1/2)*self.Ts**2*x_i.T + x_d.T.dot((G1(n) - G1(n-1)).T)).dot(self.Q).dot(self.D0_n[n]-self.Di_2n[n])
            c = (-(n-1/2)*self.Ts**2*e_s.T + (n**3-n+1/3)*self.Ts**3*x_i.T + x_d.T.dot((G3(n)-G3(n-1)).T)).dot(self.Q).dot(self.Di_1n[n])
            cf_m += a + b + c

        cf_inf = x_d.T.dot(G2(float('inf'))-G2(self.m)).dot(self.Wn[self.m-1]).dot(self.Z)

        cf = cf_m + cf_inf
        #cf = cf_m
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
        #print(results.x)
        du1 = results.x[0]
        du2 = results.x[1]
        dU = np.array([du1, du2])
        return dU
