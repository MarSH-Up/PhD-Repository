import matplotlib.pyplot as plt
from components.BilinearModel_Neurodynamics import bilinear_model_neurodynamics_z
from components.BilinearModel_Plots import *
from components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from components.Parameters import Parameters


# Event handler function to close all the plots if "escape" key is pressed
def on_key(event):
    if event.key == "escape":
        plt.close("all")
    if event.key == "escape":
        plt.close("all")


# Main fNIRS processing function
def fNIRS_Process():
    """
    Process the fNIRS data.

    Returns:
        U_stimulus: Stimulus signal
        timestamps: Array of timestamps
        Z: Neurodynamics
        dq, dh: Derivatives of blood volume and deoxyhemoglobin concentration
        Y: Optics output
    """

    # Generate stimulus train
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
    )
    Z = bilinear_model_neurodynamics_z(
        Parameters["A"],
        Parameters["B"],
        Parameters["C"],
        U_stimulus,
        Parameters["step"],
    )

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    plot_Stimulus(U_stimulus, timestamps, fig1, ax1)
    plot_neurodynamics(Z, timestamps, fig2, ax2)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    # Adjusting the layout of the plots and displaying them
    plt.tight_layout()
    plt.show()


# If this script is run directly (not imported), execute the main function
if __name__ == "__main__":
    main()


# https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b
