import numpy as np


def bilinear_model_stimulus_train_generator(
    freq, action_time, rest_time, cycles, nRegions
):
    """
    Generate a stimulus train for a bilinear model across multiple brain regions.

    Parameters:
    - freq: Sampling frequency.
    - action_time: Duration of the active period for the stimulus in seconds.
    - rest_time: Duration of the rest period between stimuli in seconds.
    - cycles: Number of stimulus cycles.
    - nRegions: Number of brain regions.

    Returns:
    - U: Stimulus matrix. Shape: (nRegions, total number of samples).
    - timestamps: Array of time points for each sample.
    """

    # Calculate number of samples for each period (activation and rest)
    activation_samples = int(action_time * freq)
    rest_samples = int(rest_time * freq)
    cycle_samples = activation_samples + rest_samples

    # Initialize U with zeros; represents absence of stimulus
    U = np.zeros((nRegions, cycle_samples * cycles))

    # Fill U with pulses (value 1) representing stimulus activation for each cycle
    for i in range(cycles):
        U[:, i * cycle_samples : i * cycle_samples + activation_samples] = 1

    # Create timestamps for the entire period
    Time_period = (action_time + rest_time) * cycles
    timestamps = np.arange(0, Time_period, 1 / freq)

    return U, timestamps
