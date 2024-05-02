import librosa.feature
from RhythmRecognition.tempo.tempogram import Tempogram
from RhythmRecognition.constants import *
import numpy as np


class AutocorrelationTempogram(Tempogram):
    """Class for computing the autocorrelation tempogram.

    The **autocorrelation tempogram** is based on autocorrelation, which  measures the
    similarity between a signal and a time-shifted version of it. If we apply autocorrelation locally
    (short-time autocorrelation) to the novelty function of the audio signal, we can then compute the
    autocorrelation tempogram, which will reveal dominant tempi."""

    def __init__(self, novelty_function: np.ndarray, similarity: int = 5, number_of_dominant_values: int = 5,
                 lower_bound: int = 40, upper_bound: int = 200):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param similarity: BPM tolerance that specifies which BPM values belong to the same group.
        :param number_of_dominant_values: How many dominant BPM values should be extracted for later computations.
        :param lower_bound: Lowest possible BPM value that will be considered.
        :param upper_bound: Highest possible BPM value that will be considered.
        """
        super().__init__(novelty_function, similarity, number_of_dominant_values, lower_bound, upper_bound)

    def _time_lag(self) -> None:
        """Compute the time-lag representation."""
        log_novelty = np.log1p(self.novelty_function)
        self.time_lag = librosa.autocorrelate(log_novelty)
        self.time_lag /= max(self.time_lag)  # normalization
        self.time_lag = self.time_lag[1:]  # starting from index 1 so we don't divide by 0 in _time_tempo

    def _time_tempo(self) -> None:
        """Compute the time-tempo representation to get possible BPM values."""
        frames = np.arange(len(self.novelty_function))
        t = librosa.frames_to_time(frames, sr=SAMPLING_RATE)
        self.bpm_values = 60 / t[1:]  # starting from index 1 so we don't divide by 0

    def _analyze_tempo(self) -> None:
        super()._analyze_tempo()

        self._time_lag()
        self._time_tempo()
        max_bpms = self._get_possible_BPM_from_tempogram()  # get first few dominant values
        self.tempo = self._find_dominant_BPM(max_bpms)

    def _compute_tempogram(self) -> None:
        self.tempogram = librosa.feature.tempogram(onset_envelope=self.novelty_function,
                                                   sr=SAMPLING_RATE,
                                                   hop_length=HOP_LENGTH,
                                                   win_length=FRAME_LENGTH)
