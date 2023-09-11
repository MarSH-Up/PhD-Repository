import numpy as np


def bilinear_model_neurodynamics_z(A, B, C, U, step):
    nRegions = A.shape[0]
    M = U.shape[0]
    simulationLength = U.shape[1]

    if M != B.shape[2]:
        raise ValueError("Unexpected number of induced connectivity parameters B.")

    Z0 = np.array([0, 0])
    Z = np.empty((nRegions, simulationLength))
    Z[:, 0] = Z0

    for t in range(1, simulationLength):
        T = np.zeros((nRegions, nRegions))
        for uu in range(M):
            tmp = U[uu, t] * B[:, :, uu]
            T = -0.5 * np.exp(T + tmp)

        SI = np.diag(A)
        A = A - np.diag(np.exp(SI) / 2 + SI)
        Zdot = (A + T) @ Z[:, t - 1] + C @ U[:, t - 1]
        Z[:, t] = Z[:, t - 1] + step * Zdot

    return Z
