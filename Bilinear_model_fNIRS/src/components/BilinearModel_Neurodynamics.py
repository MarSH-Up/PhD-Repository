import numpy as np


def Neurodynamics_Jt(A, B, U):
    nRegions = A.shape[0]

    for t in range(U.shape[1]):
        T = np.zeros(nRegions)
        for uu in range(U.shape[0]):
            tmp = U[uu, t] * B[:, :, uu]
            T = -0.5 * np.exp(T + tmp)

    SI = np.diag(A)
    new_diag = np.exp(SI) / 2 + SI
    A -= np.diagflat(new_diag)
    return A + T
