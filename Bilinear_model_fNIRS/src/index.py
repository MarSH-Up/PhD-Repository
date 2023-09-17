import matplotlib.pyplot as plt
from components.BilinearModel_Hemodynamics import Hemodynamics
from components.BilinearModel_Neurodynamics import *
from components.BilinearModel_Optics import BilinearModel_Optics
from components.BilinearModel_Plots import *
from components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from components.Parameters_case5 import Parameters


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
    Y = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"], timestamps)
    return U_stimulus, timestamps, Z, qj, pj


def main():
    U_stimulus, timestamps, Z, qj, pj = fNIRS_Process()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

    plot_Stimulus(U_stimulus, timestamps, fig, ax1)
    plot_neurodynamics(Z, timestamps, fig, ax2)
    plot_DHDQ(qj, pj, timestamps, fig, ax3)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b
