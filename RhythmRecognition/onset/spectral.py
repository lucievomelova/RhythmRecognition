import librosa
import librosa.display
from RhythmRecognition.constants import *
from RhythmRecognition.onset.novelty_function import NoveltyFunction
from RhythmRecognition.math import *


class SpectralNovelty(NoveltyFunction):
    """Class for computing spectral novelty function.

     The spectral-based novelty function is computed from time-frequency representation of the
     signal. The idea is that by tracking changes in frequency content of the signal, we can detect note onsets.
    """

    neighborhood_size_sec: float
    """Neighborhood size for local normalization in seconds."""

    neighborhood_size_frames: int
    """Neighborhood size for local normalization in frames."""
    def __init__(self, audiofile: str,
                 duration: float | None = None,
                 gamma: int = 10,
                 neighborhood_size_sec: float = 0.1,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param audiofile: filename with the song to be analyzed
        :param duration: duration of the song in seconds, if the whole song shouldn't be used
        :param gamma: compression factor (for logarithmic compression)
        :param neighborhood_size_sec: neighborhood size in seconds for local average computation
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """

        super().__init__(audiofile, duration, gamma, sampling_rate, hop_length, frame_length)

        self.neighborhood_size_sec = neighborhood_size_sec
        self.neighborhood_size_frames = librosa.time_to_frames(neighborhood_size_sec,
                                                               sr=self.sampling_rate,
                                                               hop_length=self.hop_length)

    def __compute_local_avg(self) -> np.ndarray:
        """Compute local average based on specified neighborhood_size_sec.
        \n
        To enhance the properties of the novelty function, we can subtract the local average and normalize
        the resulting function. This is a postprocessing step that should enhance the peak structure
        of the novelty function, while suppressing small fluctuations."""

        if self.len_frames is None:
            self.len_frames = len(self.novelty_function)

        local_avg = np.zeros(self.len_frames)
        for m in range(self.len_frames):
            a = max(m - self.neighborhood_size_frames, 0)
            b = min(m + self.neighborhood_size_frames + 1, self.len_frames)
            local_avg[m] = (1 / (2 * self.neighborhood_size_frames + 1)) * np.sum(self.novelty_function[a:b])
        return local_avg

    def __local_normalization(self) -> None:
        """Local normalization of the novelty function."""
        local_avg = self.__compute_local_avg()  # compute local average
        self.novelty_function = self.novelty_function - local_avg  # subtract local average
        self.novelty_function[self.novelty_function < 0] = 0  # drop negative values
        self.novelty_function = self.novelty_function / max(self.novelty_function)  # normalization

    def _compute(self) -> None:
        stft_signal = librosa.stft(self.signal, n_fft=self.frame_length,
                                   hop_length=self.hop_length, win_length=self.frame_length)

        log_comp = librosa.amplitude_to_db(1 + self.gamma * np.abs(stft_signal))  # logarithmic compression
        spec_diff = first_order_diff(log_comp)  # First-order difference
        spec_diff = half_wave_rectification(spec_diff)  # Half-wave rectification

        self.novelty_function = np.sum(spec_diff, axis=0)  # sum up
        self.novelty_function = np.concatenate((self.novelty_function, np.array([0])))
        self.__local_normalization()  # normalize
