import numpy as np

"""File with mathematical functions used in audio signal processing."""


def logarithmic_compression(x: np.ndarray, gamma: int) -> np.ndarray:
    """Logarithmic compression: L(x) = log(1+gamma*x).

    :param x: Function to compress.
    :param gamma: Compression factor.
    """
    return np.log1p(gamma * x)


def first_order_diff(x: np.ndarray) -> np.ndarray:
    """First-order diference (or discrete derivative) - calculate difference between two subsequent energy values"""
    return np.diff(x)


def half_wave_rectification(x: np.ndarray) -> np.ndarray:
    """Half-wave rectification - keep only positive values."""
    x[x < 0] = 0
    return x


def rmse(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute root-mean-square energy of the signal."""
    energy_array = []

    for i in range(0, len(x), hop_length):
        rmse_current_frame = np.sqrt(np.sum(x[i:i + frame_length] ** 2) / frame_length)
        energy_array.append(rmse_current_frame)
    energy_array = np.array(energy_array)
    return energy_array
