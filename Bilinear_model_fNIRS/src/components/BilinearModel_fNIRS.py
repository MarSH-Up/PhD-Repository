import numpy as np
from BilinearModel_Hemodynamics import Hemodynamics
from BilinearModel_Neurodynamics_v1 import Neurodynamics
from BilinearModel_Optics import BilinearModel_Optics
from BilinearModel_StimulusGenerator import *

P_SD = np.array(
    [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
)


def fNIRS_Process(Parameters):
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
        Parameters["A"].shape[0],
    )

    # Initialize the state of the neurodynamics
    Z0 = np.zeros([Parameters["A"].shape[0]])

    # Compute the neurodynamics of the system
    Z = Neurodynamics(
        Z0, timestamps, Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus
    )

    # Process hemodynamics
    qj, pj = Hemodynamics(Z.T, P_SD, Parameters["step"])

    # Process optics
    Y, dq, dh = BilinearModel_Optics(pj, qj, U_stimulus, Parameters["A"])

    return U_stimulus, timestamps, Z, dq, dh, Y
