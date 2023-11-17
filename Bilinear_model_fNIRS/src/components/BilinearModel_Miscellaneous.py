import numpy as np


def pink_noise(n_points):
    """Generates pink (1/f) noise."""
    has_uneven_points = n_points % 2
    random_complex = np.random.randn(
        n_points // 2 + 1 + has_uneven_points
    ) + 1j * np.random.randn(n_points // 2 + 1 + has_uneven_points)
    frequency_scale = np.sqrt(
        np.arange(len(random_complex)) + 1.0
    )  # Avoid divide by zero
    y = (np.fft.irfft(random_complex / frequency_scale)).real
    if has_uneven_points:
        y = y[:-1]
    return y


def generate_white_noise(timestamps, amplitude=1.0):
    """Generate white noise at given timestamps."""
    return np.random.normal(0, amplitude, len(timestamps))


def max_amplitude(signal):
    """
    Calculate the maximum amplitude of a signal.

    Parameters:
    signal (array-like): The input signal.

    Returns:
    float: The maximum amplitude of the signal.
    """
    return np.max(np.abs(signal))


def calculate_gain(percent_error, max_amplitude):
    """Calculates the gain based on percent error and max amplitude of the reference signal."""
    return (max_amplitude / 100) * percent_error


def physiologicalNoise_gains(noise_types, percent_error, max_amplitude):
    """Applies uniform gain to all noise types based on the reference signal."""
    gain = calculate_gain(percent_error, max_amplitude)
    return [gain for _ in noise_types]
