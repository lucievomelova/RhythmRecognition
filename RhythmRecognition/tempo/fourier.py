import librosa.display
from RhythmRecognition.tempo.tempogram import Tempogram
from RhythmRecognition.constants import *
import numpy as np


class FourierTempogram(Tempogram):
    """Class for computing the Fourier tempogram.
    The **Fourier tempogram** is computed by applying Short-time Fourier transform (STFT) on
    a novelty function computed from the audio signal."""

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

    def _analyze_tempo(self) -> None:
        super()._analyze_tempo()

        self.bpm_values = librosa.fourier_tempo_frequencies(sr=SAMPLING_RATE,
                                                            win_length=FRAME_LENGTH*2,
                                                            hop_length=HOP_LENGTH)
        max_bpms = self._get_possible_BPM_from_tempogram()  # get first few dominant values
        self.tempo = self._find_dominant_BPM(max_bpms)

    def _compute_tempogram(self) -> None:
        stft_spectral = librosa.stft(self.novelty_function, n_fft=FRAME_LENGTH*2, hop_length=1,
                                     win_length=FRAME_LENGTH)

        self.tempogram = np.abs(stft_spectral)


