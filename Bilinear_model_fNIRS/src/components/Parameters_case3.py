import numpy as np

A = np.array([[-0.16, -0.49, -0.25], [-0.02, -0.33, -0.15], [-0.1, -0.2, -0.3]])


B1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


B2 = np.array([[-0.02, -1, -0.5], [0, -1.31, -0.6], [0.1, -0.7, -1.2]])

B = np.zeros((3, 3, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

C = np.array([[0.08, 0, 0], [0.06, 0, 0], [0.089, 0, 0]])

freq = 10
step = 1 / freq

actionTime = 5
restTime = 25
cycles = 3

Parameters = {
    "A": A,
    "B": B,
    "C": C,
    "freq": freq,
    "step": step,
    "actionTime": actionTime,
    "restTime": restTime,
    "cycles": cycles,
}
