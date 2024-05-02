import numpy as np
import librosa
from RhythmRecognition.constants import *


class NoveltyFunction:
    """Base class for specific novelty function implementations.
    \n
    **Novelty function** is a function that
    denotes local changes in signal properties. When computed from certain signal
    properties, peaks in novelty function should indicate note onsets"""

    signal: np.ndarray
    """Audio time series."""

    duration: float
    """Duration of the song in seconds."""

    len_frames: int | None
    """Length of novelty function in frames."""

    gamma: int
    """Compression factor for logarithmic compression."""

    def __init__(self, audiofile: str, duration: float | None = None, gamma: int = 10):
        """
        :param audiofile: Name of the audio file to be processed.
        :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
        :param gamma: Compression factor for logarithmic compression.
        """

        # set duration to whole song duration if not specified
        if duration is None:
            self.signal, sr = librosa.load(audiofile, sr=SAMPLING_RATE)  # load the audiofile
            self.duration = librosa.get_duration(y=self.signal, sr=SAMPLING_RATE)
        else:
            self.signal, sr = librosa.load(audiofile, duration=duration, sr=SAMPLING_RATE)  # load the audiofile
            self.duration = duration
        self.novelty_function = None
        self.len_frames = None
        self.gamma = gamma

    def _compute(self):
        """Compute novelty function of the input signal."""
        pass

    def _get_length_in_frames(self) -> int:
        """Get length of novelty function in frames."""
        if self.len_frames is None and self.novelty_function is not None:
            self.len_frames = len(self.novelty_function)
        return self.len_frames

    def get(self) -> np.ndarray:
        """Get novelty function of input signal."""
        if self.novelty_function is None:
            self._compute()
        return self.novelty_function
