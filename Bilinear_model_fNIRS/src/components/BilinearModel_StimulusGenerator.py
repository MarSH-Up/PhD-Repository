import numpy as np


def bilinear_model_stimulus_train_generator(
    freq, action_time, rest_time, cycles, nRegions
):
    # Calculate number of samples for each period
    activation_samples = int(action_time * freq)
    rest_samples = int(rest_time * freq)
    cycle_samples = activation_samples + rest_samples

    # Initialize U
    U = np.zeros((nRegions, cycle_samples * cycles))

    # Fill U with pulses for each cycle
    for i in range(cycles):
        U[:, i * cycle_samples : i * cycle_samples + activation_samples] = 1

    # Create timestamps
    Time_period = (action_time + rest_time) * cycles
    timestamps = np.arange(0, Time_period, 1 / freq)
    return U, timestamps
