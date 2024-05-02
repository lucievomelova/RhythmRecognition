import numpy as np
from RhythmRecognition.constants import *
from RhythmRecognition.onset.novelty_function import NoveltyFunction
from RhythmRecognition.math import *


class EnergyNovelty(NoveltyFunction):
    """Class for computing energy novelty function based on root-mean-square energy (RMSE).

    **RMSE** is a time domain feature that contains information about the overall intensity or
    strength of an audio signal. This approach is based on the assumption that note
    onsets correspond with sudden increases in energy."""

    rmse: np.ndarray
    """Root-mean-square energy of the input signal."""

    def __init__(self, audiofile, duration=None, gamma=10):
        """
        :param audiofile: Name of the audio file to be processed.
        :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
        :param gamma: Compression factor for logarithmic compression.
        """

        super().__init__(audiofile, duration, gamma)

    def __compute_rmse(self) -> None:
        """Compute root-mean-square energy of the input signal."""
        self.rmse = rmse(self.signal, FRAME_LENGTH, HOP_LENGTH)

    def _compute(self) -> None:
        self.__compute_rmse()
        log_rmse = logarithmic_compression(self.rmse, self.gamma)  # logarithmic compression
        log_rmse_diff = first_order_diff(log_rmse)  # first order difference
        self.novelty_function = half_wave_rectification(log_rmse_diff)
