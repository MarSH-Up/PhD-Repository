import numpy as np


def Hemodynamics(Z, P_SD, Step):
    # Simulation size definition
    nRegions, simulationLength = Z.shape

    # Vasodilatory Signal variable
    Sj = np.full((nRegions, simulationLength), np.nan)
    Sj[:, 0] = np.zeros(nRegions)

    # Rate of blood volume
    Vj = np.zeros((nRegions, simulationLength))
    Vj[:, 0] = np.exp(np.zeros(nRegions))

    # HbT concentration (Rate)
    pj = np.full((nRegions, simulationLength), np.nan)
    pj[:, 0] = np.exp(np.zeros(nRegions))

    # HbR concentration (Rate)
    qj = np.full((nRegions, simulationLength), np.nan)
    qj[:, 0] = np.exp(np.zeros(nRegions))

    # Inflow
    fjin = np.full((nRegions, simulationLength), np.nan)
    fjin[:, 0] = np.exp(np.zeros(nRegions))
    fjout_s = np.full((nRegions, simulationLength), np.nan)
    fjout_s1 = np.full((nRegions, simulationLength), np.nan)

    # Hemodynamic parameters
    H = [0.64, 0.32, 2.00, 0.32, 0.32, 2.00]

    # Extract parameters for each region and broadcast across all regions

    Kj = H[0] * np.exp(np.concatenate(([0], [P_SD[0, 0] for _ in range(nRegions - 1)])))
    Yj = H[1] * np.exp(np.concatenate(([0], [P_SD[1, 0] for _ in range(nRegions - 1)])))
    Tj = H[2] * np.exp(np.concatenate(([0], [P_SD[2, 0] for _ in range(nRegions - 1)])))
    Tjv = H[5] * np.exp(
        np.concatenate(([0], [P_SD[3, 0] for _ in range(nRegions - 1)]))
    )
    phi = H[3]

    for t in range(1, simulationLength):
        Sj_dot = Z[:, t - 1] - Kj * Sj[:, t - 1] - Yj * (fjin[:, t - 1] - 1)
        fjin_dot = Sj[:, t - 1]

        fv_s = Vj[:, t - 1] ** (1 / phi)
        Vj_dot = (fjin[:, t - 1] - fv_s) / (Tj * Tjv * Vj[:, t - 1])

        fjout = fv_s + Tjv * Vj_dot
        Efp = (1 - (1 - H[4]) ** (1 / fjin[:, t - 1])) / H[4]

        qj_dot = ((fjin[:, t - 1] * Efp - fjout * qj[:, t - 1]) / Vj[:, t - 1]) / (
            Tj * qj[:, t - 1]
        )
        pj_dot = (fjin[:, t - 1] - (fjout * pj[:, t - 1]) / Vj[:, t - 1]) / Tj

        # Euler steps
        Sj[:, t] = Sj[:, t - 1] + Step * Sj_dot
        Vj[:, t] = Vj[:, t - 1] + Step * Vj_dot
        fjin[:, t] = fjin[:, t - 1] + Step * fjin_dot
        qj[:, t] = qj[:, t - 1] + Step * qj_dot
        pj[:, t] = pj[:, t - 1] + Step * pj_dot
        fjout_s[:, t] = fjout
        fjout_s1[:, t] = Efp

    return qj, pj
