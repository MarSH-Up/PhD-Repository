import numpy as np

# New A matrix for 5 regions
A = np.array(
    [
        [-0.16, -0.49, -0.25, -0.12, -0.18],
        [-0.02, -0.33, -0.15, -0.09, -0.21],
        [-0.1, -0.2, -0.3, -0.15, -0.24],
        [-0.15, -0.27, -0.19, -0.3, -0.2],
        [-0.11, -0.23, -0.18, -0.17, -0.29],
    ]
)

# New B1 matrix for 5 regions
B1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

# New B2 matrix for 5 regions
B2 = np.array(
    [
        [-0.02, -1, -0.5, -0.6, -0.7],
        [0, -1.31, -0.6, -0.5, -0.8],
        [0.1, -0.7, -1.2, -0.9, -0.6],
        [0.05, -0.8, -0.9, -1, -0.7],
        [0.03, -0.6, -0.7, -0.8, -1.1],
    ]
)

# New B tensor for 5 regions
B = np.zeros((5, 5, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

# New C matrix for 5 regions
C = np.array(
    [
        [0.08, 0, 0, 0, 0],
        [0, 0.06, 0, 0, 0],
        [0, 0, 0.07, 0, 0],
        [0, 0, 0, 0.09, 0],
        [0, 0, 0, 0, 0.01],
    ]
)

# New P_SD matrix for 5 regions (and 2 states as in your original matrix)
P_SD = np.array(
    [
        [0.0775, -0.0087],
        [-0.1066, 0.0299],
        [0.0440, -0.0129],
        [0.8043, -0.7577],
        [0.0950, -0.0650],
    ]
)


freq = 10.84
step = 1 / freq

actionTime = [5, 5, 5, 5, 5]
restTime = [25, 25, 25, 25, 25]
cycles = [
    2,
    0,
    0,
    0,
    0,
]


Parameters = {
    "A": A,
    "B": B,
    "C": C,
    "P_SD": P_SD,
    "freq": freq,
    "step": step,
    "actionTime": actionTime,
    "restTime": restTime,
    "cycles": cycles,
}
