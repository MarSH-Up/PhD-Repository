import matplotlib.pyplot as plt
from components.BilinearModel_Hemodynamics import Hemodynamics
from components.BilinearModel_Neurodynamics import *
from components.BilinearModel_Optics import BilinearModel_Optics
from components.BilinearModel_Plots import *
from components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from components.Parameters_case5 import Parameters

from Bilinear_model_fNIRS.src.components.BilinerModel_Noises import awgn


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
    U_stimulus, timestamps, Z, qj, pj, Y = fNIRS_Process()

    # Initialize the plotting layout
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))

    # Plotting functions from BilinearModel_Plots
    plot_Stimulus(U_stimulus, timestamps, fig, ax1)
    plot_neurodynamics(Z, timestamps, fig, ax2)
    plot_DHDQ(qj, pj, timestamps, fig, ax3)
    plot_Y(Y, timestamps, fig, ax4)

    # Adding noise to the signal and plotting it
    noisy_signal = awgn(Y, 5, "measured")
    plot_Y(noisy_signal, timestamps, fig, ax5)

    # Binding the on_key event function to the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Adjusting the layout of the plots and displaying them
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
