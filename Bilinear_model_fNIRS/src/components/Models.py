import numpy as np
from scipy.integrate import odeint


def Neurodynamics_Z(Z, t, A, B, C, U):
    nRegions = A.shape[0]
    index = min(int(t * 10), U.shape[1] - 1)

    T = np.zeros(nRegions)
    for uu in range(B.shape[2]):
        tmp = U[uu, index] * B[:, :, uu]
        T = -0.5 * np.exp(T + tmp)

    SI = np.diag(A)
    new_diag = np.exp(SI) / 2 + SI
    A -= np.diagflat(new_diag)
    J_t = A + T

    dZdt = np.dot(J_t, Z) + np.dot(C, U[:, index])
    return dZdt
