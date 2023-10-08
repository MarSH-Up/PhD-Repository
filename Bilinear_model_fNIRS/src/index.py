# Importing necessary libraries
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

# Define paths to easily import custom modules.
# Assuming the script is located two directories deep from the root of the project.
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))

# Adding the root directory to the system path
sys.path.append(root_directory)

# Importing components of the Bilinear_model_fNIRS project
from Bilinear_model_fNIRS.src.components.BilinearModel_Hemodynamics import Hemodynamics
from Bilinear_model_fNIRS.src.components.BilinearModel_Neurodynamics import (
    Neurodynamics,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Optics import (
    BilinearModel_Optics,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Plots import *
from Bilinear_model_fNIRS.src.components.BilinearModel_StimulusGenerator import *
from Bilinear_model_fNIRS.src.components.BilinerModel_Noises import awgn
from Bilinear_model_fNIRS.src.components.Parameters.Parameters_case3 import Parameters


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

    """
    # Define parameters
    freq = 10  # Sampling frequency
    nRegions = 3  # Number of brain regions

    # Different action times, rest times, and cycles for each region
    action_times = [0, 3, 5]  # in seconds
    rest_times = [0, 27, 25]  # in seconds
    cycles_list = [0, 2, 3]  # Number of cycles

    # Generate the stimulus train
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        freq, action_times, rest_times, cycles_list, nRegions
    )

    # Generate stimulus train
   """
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator_constant(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
        Parameters["A"].shape[0],
    )

    # Initialize the state of the neurodynamics
    Z0 = np.zeros([Parameters["A"].shape[0]])

    # Compute the neurodynamics of the system
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )

    # Process hemodynamics
    qj, pj = Hemodynamics(Z.T, Parameters["P_SD"], Parameters["step"])

    # Process optics
    Y, dq, dh = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"])

    return U_stimulus, timestamps, Z, dq, dh, Y


# Main function that calls fNIRS_Process and then plots the results
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


# If this script is run directly (not imported), execute the main function
if __name__ == "__main__":
    main()


# https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b
