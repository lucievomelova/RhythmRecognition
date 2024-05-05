import librosa.display
from RhythmRecognition.tempo.tempogram import Tempogram
from RhythmRecognition.tempo.fourier import FourierTempogram
from RhythmRecognition.tempo.autocorrelation import AutocorrelationTempogram
from RhythmRecognition.constants import *
import numpy as np


class HybridTempogram(Tempogram):
    """Class for computing the hybrid tempogram.

    The **hybrid tempogram** combines the Fourier and autocorrelation tempograms. It is
    a simple element-wise product of both tempograms with some added post-processing for
    enhancing dominant tempi."""

    gamma: int
    """Compression factor for logarithmic compression."""

    def __init__(self, novelty_function: np.ndarray,
                 similarity: int = 5,
                 number_of_dominant_values: int = 5,
                 lower_bound: int = 40,
                 upper_bound: int = 200,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param similarity: BPM tolerance that specifies which BPM values belong to the same group.
        :param number_of_dominant_values: How many dominant BPM values should be extracted for later computations.
        :param lower_bound: Lowest possible BPM value that will be considered.
        :param upper_bound: Highest possible BPM value that will be considered.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """
        super().__init__(novelty_function, similarity, number_of_dominant_values, lower_bound, upper_bound,
                         sampling_rate, hop_length, frame_length)
        self.gamma = 5

    def _analyze_tempo(self) -> None:
        super()._analyze_tempo()

        self.bpm_values = librosa.fourier_tempo_frequencies(sr=self.sampling_rate,
                                                            win_length=self.frame_length*2,
                                                            hop_length=self.hop_length)
        max_bpms = self._get_possible_BPM_from_tempogram()  # get first few dominant values
        self.tempo = self._find_dominant_BPM(max_bpms)

    def _compute_tempogram(self) -> None:
        fourier_tempogram = FourierTempogram(self.novelty_function[1:])
        autocorrelation_tempogram = AutocorrelationTempogram(self.novelty_function)
        fourier_tempogram = fourier_tempogram.get_tempogram()
        autocorrelation_tempogram = autocorrelation_tempogram.get_tempogram()

        self.tempogram = fourier_tempogram[1:] * autocorrelation_tempogram
        self.tempogram = np.log1p(self.gamma * self.tempogram)
