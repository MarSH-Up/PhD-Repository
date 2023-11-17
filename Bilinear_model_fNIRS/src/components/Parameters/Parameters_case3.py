import numpy as np

A = np.array([[-0.16, -0.49, -0.25], [-0.02, -0.33, -0.15], [-0.1, -0.2, -0.3]])


B1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


B2 = np.array([[-0.02, -1, -0.5], [0, -1.31, -0.6], [0.1, -0.7, -1.2]])

B = np.zeros((3, 3, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2


P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)


C = np.array([[0.08, 0, 0], [0, 0.06, 0], [0, 0, 0.07]])

freq = 10.84
step = 1 / freq

actionTime = [5, 5, 5]
restTime = [25, 25, 25]
cycles = [5, 5, 5]

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
