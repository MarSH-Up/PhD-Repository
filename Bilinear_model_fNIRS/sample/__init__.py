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
from Bilinear_model_fNIRS.src.components.BilinearModel_Plots import (
    plot_hemodynamics,
    plot_neurodynamics,
    plot_Stimulus,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
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
    print(timestamps.shape)
    Z0 = np.zeros([Parameters["A"].shape[0]])
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )
    qj, pj = Hemodynamics(Z.T, Parameters["P_SD"], Parameters["step"])
    Y = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"], timestamps)
    print(qj.shape, pj.shape)

    return U_stimulus, timestamps, Z, qj, pj


def main():
    U_stimulus, timestamps, Z, qj, pj = fNIRS_Process()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    plot_Stimulus(U_stimulus, timestamps, fig1, ax1)
    plot_neurodynamics(Z, timestamps, fig2, ax2)
    plot_hemodynamics(qj, pj, timestamps, fig3, ax3)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)
    fig3.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
