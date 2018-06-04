# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:07:03 2017

@author: Igor Yamamoto

Alterado em 07/04/2018 - MLima
"""
import numpy as np
from scipy import signal
from scipy.linalg import block_diag

class TransferFunctionDelay(signal.TransferFunction):
    def __init__(self, num, den, delay=0):
        super().__init__(num, den)
        self.delay = delay
        
    def __repr__(self):
        return "num={}\nden={}\ndt={}\ndelay={}".format(self.num, self.den, self.dt, self.delay)

class OPOM(object):
    def __init__(self, H, Ts):
        self.H = np.array(H)
        if self.H.size == 1:
            self.ny = 1
            self.nu = 1
        else:
            self.ny = self.H.shape[0]
            self.nu = self.H.shape[1]
        self.Ts = Ts
        self.na = self._max_order()  # max order of Gij
        self.nd = self.ny*self.nu*self.na
        self.delay_matrix = self._delay_matrix()
        self.theta_max = self.delay_matrix.max()
        self.nz = self.theta_max*self.nu
        self.nx = 2*self.ny+self.nd+self.nz
        self.X = np.zeros(self.nx)
        self.R, self.D0, self.Di, self.Dd, self.F, self.N, self.Istar = self._create_matrices()
        self.A, self.B, self.C, self.D = self._create_state_space()
        self.Psi = self.Psi(Ts)

    def __repr__(self):
        return "A=\n%s\n\nB=\n%s\n\nC=\n%s\n\nD=\n%s" % (self.A.__repr__(),
                                                         self.B.__repr__(),
                                                         self.C.__repr__(),
                                                         self.D.__repr__())
        
    def _delay_matrix(self):
        return np.apply_along_axis(
                lambda row: list(map(lambda tf: tf.delay, row)),
                0,
                self.H)
        
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
        Istar = np.array([])
        for i in range(self.ny):
            d0_x = np.array([])
            di_x = np.array([])
            for j in range(self.nu):
                if self.H.size > 1:
                    b = self.H[i][j].num
                    a = self.H[i][j].den
                else:
                    b = self.H[i].num
                    a = self.H[i].den
                d0, dd, di, r = self._get_coeff(b, a)
                d0_x = np.hstack((d0_x, d0))
                Dd = np.append(Dd, dd)
                di_x = np.hstack((di_x, di))
                R = np.append(R, r)
            D0 = np.vstack((D0, d0_x))
            Di = np.vstack((Di, di_x))
            tem_i = int(not all(R))
            Istar = np.append(Istar,tem_i)
        Dd = np.diag(Dd)
        D0 = D0[1:]
        Di = Di[1:]
        Istar = np.diag(Istar)

        # Define matrices F[ndxnd], J[nu.naxnu], N[ndxnu]
        F = np.diag(np.exp(self.Ts*R))

        J = np.zeros((self.nu*self.na, self.nu))
        for col in range(self.nu):
            J[col*self.na:col*self.na+self.na, col] = np.ones(self.na)

        N = J
        for _ in range(self.ny-1):
            N = np.vstack((N, J))

        return R, D0, Di, Dd, F, N, Istar

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

    def Bs(self, l):
        return np.where(self.delay_matrix==l,                    
                        self.D0 + self.Ts*self.Di,
                        0)

    def Bi(self, l):
        return np.where(self.delay_matrix==l,                    
                        self.Di,
                        0)

    def Bd(self, l):
        Bd = self.Dd.dot(self.F).dot(self.N)
        flat_delay_matrix = self.delay_matrix.flatten().tolist()
        delay_matrix_nd = list(map(lambda x: [x]*self.na,
                                   flat_delay_matrix))
        delay_matrix_nd_nu = np.diag(
                                np.array(delay_matrix_nd).flatten()
                             ).dot(self.N)
        return np.where(delay_matrix_nd_nu==l,                    
                        Bd,
                        0)
        
    def _create_Az(self):
        z1_row = np.zeros((self.nu, self.nx))
        
        if self.theta_max == 1:
            return z1_row
        else:
            zero_block = np.zeros(((self.theta_max-1)*self.nu, self.nx-self.nz))
            eye_diag = block_diag(*([np.eye(self.nu).tolist()]*(self.theta_max-1)))
            zero_column = np.zeros(((self.theta_max-1)*self.nu, self.nu))
            z2_to_ztheta_max_row = np.hstack((zero_block, eye_diag, zero_column))
            
            return np.vstack((z1_row, z2_to_ztheta_max_row))
    
    def _create_state_space(self):
        a1 = np.hstack((np.eye(self.ny),
                        np.zeros((self.ny, self.nd)),
                        self.Ts*self.Istar))
        a2 = np.hstack((np.zeros((self.nd, self.ny)),
                        self.F,
                        np.zeros((self.nd, self.ny))))
        a3 = np.hstack((np.zeros((self.ny, self.ny)),
                        np.zeros((self.ny, self.nd)),
                        self.Istar))
        for i in range(self.theta_max):
            a1 = np.hstack((a1, self.Bs(i+1)))
            a2 = np.hstack((a2, self.Bd(i+1)))
            a3 = np.hstack((a3, self.Bi(i+1)))
        Ax = np.vstack((a1, a2, a3))
        if self.theta_max == 0:
            A = Ax
            B = np.vstack((self.Bs(0),
                           self.Bd(0),
                           self.Bi(0)))
        elif self.theta_max == 1:
            Az = self._create_Az()
            A = np.vstack((Ax, Az))
            B = np.vstack((self.Bs(0),
                           self.Bd(0),
                           self.Bi(0),
                           np.eye(self.nu)))
        else:
            Az = self._create_Az()
            A = np.vstack((Ax, Az))
            B = np.vstack((self.Bs(0),
                           self.Bd(0),
                           self.Bi(0),
                           np.eye(self.nu),
                           np.zeros(((self.theta_max-1)*self.nu, self.nu))))
        
        def C(t):
            if self.theta_max == 0:
                return np.hstack((np.eye(self.ny), self.Psi(t), np.eye(self.ny)*t))
            else:
                up = np.hstack((np.eye(self.ny),
                                self.Psi(t),
                                np.eye(self.ny)*t,
                                np.zeros((self.ny, self.nz))))
                eye_diag = block_diag(*([np.eye(self.nu).tolist()]*(self.theta_max)))
                bottom = np.hstack((np.zeros((self.nz, self.nx-self.nz)),
                                    eye_diag))
                return np.vstack((up, bottom))

        D = np.zeros((self.ny, self.nu))

        return A, B, C(0), D

    def output(self, dU, samples=1):
        try:
            shape = dU.shape[1]
            print(shape)
        except IndexError:
            dU = np.reshape(dU,(1,self.nu))
        X = np.zeros((samples+1, self.nx))
        X[0] = self.X
        Y = np.zeros((samples+1, self.ny+self.nz))
        Y[0] = self.C.dot(X[0])
        for k in range(samples):
            X[k+1] = self.A.dot(X[k]) + self.B.dot(dU[k])
            Y[k+1] = self.C.dot(X[k+1])

        self.X = X[samples]
        return X[samples], Y[samples]

if __name__ == '__main__':
    h11 = TransferFunctionDelay([-0.19], [1, 0], delay=2)
    h12 = TransferFunctionDelay([-1.7], [19.5, 1])
    h21 = TransferFunctionDelay([-0.763], [31.8, 1])
    h22 = TransferFunctionDelay([0.235], [1, 0])
    H = [[h11, h12], [h21, h22]]
    Ts = 1
    model = OPOM(H, Ts)
    
    g11 = TransferFunctionDelay([2.6], [62, 1], delay=1)
    g12 = TransferFunctionDelay([1.5], [1426, 85, 1], delay=2) # g12 = 1.5/(1+23s)(1+62s)
    g21 = TransferFunctionDelay([1.4], [2700, 120, 1], delay=3) # g21 = 1.4/(1+30s)(1+90s)
    g22 = TransferFunctionDelay([2.8], [90, 1], delay=4)
    G = [[g11, g12], [g21, g22]]
    sys = OPOM(G, Ts)