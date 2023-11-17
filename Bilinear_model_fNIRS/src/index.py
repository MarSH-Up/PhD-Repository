# Importing necessary libraries
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

# Define paths to easily import custom modules.
# Assuming the script is located two directories deep from the root of the project.
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
print(root_directory)
# Adding the root directory to the system path
sys.path.append(root_directory)

from Bilinear_model_fNIRS.src.components.BilinearModel_Hemodynamics import Hemodynamics
from Bilinear_model_fNIRS.src.components.BilinearModel_Neurodynamics_v1 import (
    Neurodynamics,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Optics import (
    calculate_hemoglobin_changes,
    compute_optical_response,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_Plots import *
from Bilinear_model_fNIRS.src.components.BilinearModel_SemisyntheticNoise import (
    add_noise_to_hemodynamics,
    plotSemiSyntheticData,
    semisynthecticDataExtraction,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from Bilinear_model_fNIRS.src.components.BilinearModel_SyntheticNoise import (
    combine_noises,
    synthetic_noise_plots,
    synthetic_physiological_noise_model,
)
from Bilinear_model_fNIRS.src.components.Parameters.Parameters import Parameters


# Event handler function to close all the plots if "escape" key is pressed
def on_key(event):
    if event.key == "escape":
        plt.close("all")
    if event.key == "escape":
        plt.close("all")


# Main fNIRS processing function
def fNIRS_Process(NoiseSelection):
    """
    Process the fNIRS data.

    Returns:
        U_stimulus: Stimulus signal
        timestamps: Array of timestamps
        Z: Neurodynamics
        dq, dh: Derivatives of blood volume and deoxyhemoglobin concentration
        Y: Optics output
    """
    print(Parameters["actionTime"])

    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
        Parameters["A"].shape[0],
    )

    # Initialize the state of the neurodynamics
    Z0 = np.zeros([Parameters["A"].shape[0]])

    # Compute the neurodynamics of the system
    print(U_stimulus)
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )

    # Process hemodynamics
    qj, pj = Hemodynamics(Z.T, Parameters["P_SD"], Parameters["step"])

    pj_noise = pj.copy()
    qj_noise = qj.copy()
    percent_error = 30

    # PhysiologicalNoise Inclusion
    if NoiseSelection == "Synthetic":
        noise_types = [
            "heart",
            "breathing",
            "vasomotion",
            "white",
        ]  # Types of noise to generate
        # Example percent error
        noises_with_gains = synthetic_physiological_noise_model(
            timestamps, noise_types, pj_noise, percent_error
        )
        # Labels corresponding to the noise types
        labels = ["Heart Rate", "Vasomotion", "Breathing Rate", "White"]

        # Plot the noises with gains
        synthetic_noise_plots(timestamps, noises_with_gains, labels)

        # Combine the noises into hemodyanmics
        combined_noises = combine_noises(noises_with_gains, pj_noise.shape[0])

        # Add the combined noise to qj and pj
        qj_noise = qj_noise + combined_noises
        pj_noise = pj_noise + combined_noises

    elif NoiseSelection == "Semisynthetic":
        semisynthetic_noises = semisynthecticDataExtraction(
            Parameters["A"].shape[0], Parameters["freq"], len(timestamps)
        )

        plotSemiSyntheticData(semisynthetic_noises)
        qj_noise, pj_noise = add_noise_to_hemodynamics(
            qj_noise, pj_noise, semisynthetic_noises, percent_error, timestamps
        )

    dq, dh = calculate_hemoglobin_changes(pj_noise, qj_noise)
    # Process optics
    # Y, dq, dh = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"])
    Y = compute_optical_response(dq, dh)

    return U_stimulus, timestamps, Z, dq, dh, Y, qj, pj


# Main function that calls fNIRS_Process and then plots the results
def main():
    U_stimulus, timestamps, Z, dq, dh, Y, qj_clean, pj_clean = fNIRS_Process(
        "Synthetic"
    )

    # Initialize the plotting layout
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))

    # Plotting functions from BilinearModel_Plots
    plot_Stimulus(U_stimulus, timestamps, fig, ax1)
    plot_neurodynamics(Z, timestamps, fig, ax2)
    plot_DHDQ(qj_clean, pj_clean, timestamps, fig, ax3)
    plot_DHDQ(dq, dh, timestamps, fig, ax4)
    plot_Y(Y, timestamps, fig, ax5)

    # Binding the on_key event function to the figure
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Adjusting the layout of the plots and displaying them
    plt.tight_layout()
    plt.show()


# If this script is run directly (not imported), execute the main function
if __name__ == "__main__":
    main()


# https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b
