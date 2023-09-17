import matplotlib.pyplot as plt
import numpy as np


def BilinearModel_Optics(pj, qj, U, A):
    nRegions = A.shape[0]
    simulationLength = U.shape[1]

    # optics parameters
    N = [0.65, 71, 2]
    P0 = N[1]
    base_hbr = N[1] * (1 - N[0])

    dq = np.zeros((nRegions, simulationLength))
    dp = np.zeros((nRegions, simulationLength))
    dh = np.zeros((nRegions, simulationLength))
    Y = np.zeros((2 * nRegions, simulationLength))

    F_P = np.array(
        [
            (0.0007358251 * 7.5, 0.001104715 * 6.5),
            (0.001159306 * 7.5, 0.0007858993 * 6.5),
        ]
    )

    for t in range(simulationLength):
        dp[:, t] = (pj[:, t] - 1) * P0
        dq[:, t] = (qj[:, t] - 1) * base_hbr
        dh[:, t] = dp[:, t] - dq[:, t]

        for r in range(nRegions):
            dhq = np.array([dq[r, t], dh[r, t]])
            Y_r = F_P @ dhq
            Y[2 * r : 2 * r + 2, t] = Y_r

    return Y, dh, dq


# You can call the function to test it:
# BilinearModel_Optics(pj, qj, U, A)
