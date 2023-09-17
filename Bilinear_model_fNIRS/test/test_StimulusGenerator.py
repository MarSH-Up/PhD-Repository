import os
import sys

import numpy as np

# Set the current and root directories to find required files/modules
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(root_directory)

# Import the bilinear_model_stimulus_train_generator function from its module
from src.components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)


def test_bilinear_model_stimulus_train_generator():
    """
    Test the function bilinear_model_stimulus_train_generator to ensure it generates
    the correct stimulus train across multiple brain regions.

    Tests cover:
    - Output shapes
    - Correct activation and rest patterns in U
    - Correct timestamps
    """
    # Define parameters for the stimulus generator
    freq = 1  # Frequency of 1Hz
    action_time = 5  # Activation period of 5 seconds
    rest_time = 100  # Rest period of 100 seconds
    cycles = 4  # Total of 4 cycles
    nRegions = 5  # Stimulus train for 5 brain regions

    # Generate stimulus train and timestamps using the function
    U, timestamps = bilinear_model_stimulus_train_generator(
        freq, action_time, rest_time, cycles, nRegions
    )

    # Verify the shape of the generated stimulus matrix (U) and timestamps
    assert U.shape == (nRegions, (action_time + rest_time) * freq * cycles)
    assert timestamps.shape == ((action_time + rest_time) * freq * cycles,)

    # Verify if the generated stimulus matrix (U) has the correct activation and rest patterns for each region
    activation_samples = int(action_time * freq)
    rest_samples = int(rest_time * freq)

    for i in range(cycles):
        # Ensure that the active period is correctly set to 1
        assert np.all(
            U[
                :,
                i
                * (activation_samples + rest_samples) : i
                * (activation_samples + rest_samples)
                + activation_samples,
            ]
            == 1
        )
        # Ensure that the rest period is correctly set to 0
        assert np.all(
            U[
                :,
                i * (activation_samples + rest_samples)
                + activation_samples : (i + 1) * (activation_samples + rest_samples),
            ]
            == 0
        )

    # Verify if the generated timestamps are correct
    expected_timestamps = np.arange(0, (action_time + rest_time) * cycles, 1 / freq)
    assert np.all(timestamps == expected_timestamps)

    print("All tests passed!")


if __name__ == "__main__":
    # Run the test function when the script is executed
    test_bilinear_model_stimulus_train_generator()
