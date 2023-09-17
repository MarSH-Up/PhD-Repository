import numpy as np

# A matrix for 5 regions
A = np.array(
    [
        [-0.16, -0.49, -0.25, -0.2, -0.15],
        [-0.02, -0.33, -0.15, -0.1, -0.05],
        [-0.1, -0.2, -0.3, -0.25, -0.2],
        [-0.05, -0.1, -0.15, -0.2, -0.25],
        [-0.1, -0.05, -0.1, -0.15, -0.2],
    ]
)

# B1 matrix for 5 regions
B1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

# B2 matrix for 5 regions
B2 = np.array(
    [
        [-0.02, -1, -0.5, -0.4, -0.3],
        [0, -1.31, -0.6, -0.5, -0.4],
        [0.1, -0.7, -1.2, -1.1, -1],
        [0.05, -0.6, -1.1, -1, -0.9],
        [0.05, -0.5, -1, -0.9, -0.8],
    ]
)

# B matrix for 5 regions
B = np.zeros((5, 5, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

# C matrix for 5 regions
C = np.array(
    [
        [0.08, 0, 0, 0, 0],
        [0.06, 0, 0, 0, 0],
        [0.05, 0, 0, 0, 0],
        [0.07, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0],
    ]
)

P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)

freq = 10
step = 1 / freq
actionTime = 10
restTime = 50
cycles = 3

Parameters = {
    "A": A,
    "B": B,
    "C": C,
    "freq": freq,
    "step": step,
    "P_SD": P_SD,
    "actionTime": actionTime,
    "restTime": restTime,
    "cycles": cycles,
}
