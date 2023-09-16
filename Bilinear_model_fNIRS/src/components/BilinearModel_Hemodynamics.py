import numpy as np
from scipy.integrate import odeint


def hemodynamics_system(Y, t, Z, U, P_SD, A, Step):
    # Extract parameters from P_SD and H
    H = [0.64, 0.32, 2.00, 0.32, 0.32, 2.00]
    Kj = H[0] * np.exp(P_SD[0, 0])
    gamma_j = H[1] * np.exp(P_SD[1, 0])
    Tj = H[2] * np.exp(P_SD[2, 0])
    Tjv = H[5] * np.exp(P_SD[3, 0])
    alpha = H[4]
    rho = 0.32  # Resting oxygen extraction fraction

    # Unpack Y for two regions
    Sj = Y[:2]
    fjin = Y[2:4]
    Vj = Y[4:6]
    Pj = Y[6:8]
    qj = Y[8:10]

    # Equations
    Zj = Z[int(t / Step)]
    Sj_dot = Zj - Kj * Sj - gamma_j * (fjin - 1)
    fjin_dot = Sj
    E = 1 - (1 - rho) ** (1 / fjin)
    fjout = Vj ** (1 / alpha) + Tjv * (fjin - Vj ** (1 / alpha) / (Tj * Tjv))
    Vj_dot = (fjin - fjout) / Tj
    Pj_dot = (fjin - fjout) * Pj / (Tj * Vj)
    qj_dot = (fjin * E / rho - fjout * qj / Vj) / Tj

    return np.concatenate([Sj_dot, fjin_dot, Vj_dot, Pj_dot, qj_dot])
