import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(root_directory)
from matplotlib import pyplot as plt

from Bilinear_model_fNIRS.src.components.BilinearModel_Hemodynamics import Hemodynamics
from Bilinear_model_fNIRS.src.components.BilinearModel_Neurodynamics import (
    Neurodynamics,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Optics import (
    BilinearModel_Optics,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Plots import *
from Bilinear_model_fNIRS.src.components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from Bilinear_model_fNIRS.src.components.BilinerModel_Noises import awgn
from Bilinear_model_fNIRS.src.components.Parameters import Parameters


def on_key(event):
    if event.key == "escape":
        plt.close("all")


def fNIRS_Process():
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
        Parameters["A"].shape[0],
    )
    Z0 = np.zeros([Parameters["A"].shape[0]])
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )
    qj, pj = Hemodynamics(Z.T, Parameters["P_SD"], Parameters["step"])
    Y, dq, dh = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"])
    return U_stimulus, timestamps, Z, dq, dh, Y


def main():
    U_stimulus, timestamps, Z, qj, pj, Y = fNIRS_Process()

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))

    plot_Stimulus(U_stimulus, timestamps, fig, ax1)
    plot_neurodynamics(Z, timestamps, fig, ax2)
    plot_DHDQ(qj, pj, timestamps, fig, ax3)
    plot_Y(Y, timestamps, fig, ax4)
    noisy_signal = awgn(Y, 5, "measured")
    plot_Y(noisy_signal, timestamps, fig, ax5)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
