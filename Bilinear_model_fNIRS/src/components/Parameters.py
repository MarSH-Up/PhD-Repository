import numpy as np


A = np.array([[-0.16, -0.49],
              [-0.02, -0.33]])

B1 = np.array([[0, 0],
               [0, 0]])
B2 = np.array([[-0.02, -1],
               [0, -1.31]])

B = np.zeros((2, 2, 2))
B[:, :, 0] = B1
B[:, :, 1] = B2

C = np.array([[0.08, 0], [0.06, 0]])

freq = 10
step = 1/freq

actionTime = 5
restTime = 25
cycles = 3

Parameters = {
    'A': A,
    'B1': B1,
    'B2': B2,
    'B': B,
    'C': C,
    'freq': freq,
    'step': step,
    'actionTime': actionTime,
    'restTime': restTime,
    'cycles': cycles
}
