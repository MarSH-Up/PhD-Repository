import os
import sys

import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(root_directory)

from src.components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)


def test_bilinear_model_stimulus_train_generator():
    freq = 1  # 1Hz
    action_time = 5  # 2 seconds
    rest_time = 100  # 3 seconds
    cycles = 4  # 4 cycles
    nRegions = 5  # 5 regions

    U, timestamps = bilinear_model_stimulus_train_generator(
        freq, action_time, rest_time, cycles, nRegions
    )

    # Check output shapes
    assert U.shape == (nRegions, (action_time + rest_time) * freq * cycles)
    assert timestamps.shape == ((action_time + rest_time) * freq * cycles,)

    # Check if U has the correct activation and rest patterns
    activation_samples = int(action_time * freq)
    rest_samples = int(rest_time * freq)

    for i in range(cycles):
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
        assert np.all(
            U[
                :,
                i * (activation_samples + rest_samples)
                + activation_samples : (i + 1) * (activation_samples + rest_samples),
            ]
            == 0
        )

    # Check if timestamps are correct
    expected_timestamps = np.arange(0, (action_time + rest_time) * cycles, 1 / freq)
    assert np.all(timestamps == expected_timestamps)

    print("All tests passed!")


if __name__ == "__main__":
    test_bilinear_model_stimulus_train_generator()
