import librosa.display
import numpy as np
from RhythmRecognition.beat.peak_picking import peak_pick
from RhythmRecognition.beat.click_track import get_click_times_sec
from RhythmRecognition.constants import *


class BeatTracker:
    """Base class for specific beat tracking approaches.
    \n
    **Beat** is specified by two parameters: the phase and the period.
    The period *p* is given by the reciprocal of the tempo.
    The phase *s* then specifies a time shift. The beat function is a sinusoid with period p shifted by s.
    \n
    Beat tracking is a complex process. First step is extracting peaks from the novelty function of the input signal.
    Then we need to compute the beat period and phase. Period is easily computed from tempo, but computing phase
    - or the time shift in other words - is more complicated. The different child classes of BeatTracker offer
    different approaches for finding time shift.
    """

    novelty_function: np.ndarray
    """Novelty function of the input audio signal."""

    len_frames: int
    """Length of novelty function in frames."""

    frame_times: np.ndarray
    """Time of each frame."""

    tempo: int
    """Tempo of the song."""

    period: float
    """Length of beat period."""

    alpha: float
    """Parameter for peak picking specifying the ratio for how many peaks minimum should be extracted
    from the novelty function. The base is (tempo/60) * duration, alpha then specifies the number by 
    which the base should be multiplied."""

    time_shift: float
    """Time shift of the beat sinusoid from the start."""

    duration: float
    """Duration of the song in seconds."""

    part_len_seconds: int
    """Length of part in seconds. Peak picking will be done on smaller parts of the song of the specified length."""

    min_delta: float
    """Minimum delta for peak picking. Peak picking algorithm slowly makes delta
    smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
    ends even before finding the minimal number of peaks."""

    click_times_sec: np.ndarray
    """Click times. They represent a metronome set to the defined tempo that is started from time 0. Beat tracking 
    works by shifting these click times and finding the best time shift so that the most click time align with 
    note onset candidates."""

    peaks: np.ndarray
    """Peaks in novelty function (their frame position). Peaks represent note onset candidates."""

    peak_times: np.ndarray
    """Peak times in seconds."""

    sampling_rate: int
    """Defines the number of samples per second taken from a continuous signal to make a discrete signal."""

    frame_length: int
    """Number of samples in a frame."""

    hop_length: int
    """Number of samples by which we have to advance between two consecutive frames."""

    def __init__(self,
                 novelty_function: np.ndarray,
                 tempo: int,
                 duration: float | None,
                 alpha: float = 1.5,
                 part_len_seconds: int = 10,
                 min_delta: int = 0.00001,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param tempo: Tempo (in BPM) of the input song.
        :param duration: Duration of the input song in seconds.
        :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
            from the novelty function. The base is (tempo/60) * duration.
        :param part_len_seconds: Peak picking is done on smaller parts of the song. This parameter specifies
            the part length in seconds.
        :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
            smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
            ends even before finding the desired number of peaks.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """
        self.novelty_function = novelty_function
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.len_frames = len(self.novelty_function)
        self.frame_times = librosa.frames_to_time(np.arange(len(self.novelty_function)),
                                                  sr=self.sampling_rate, hop_length=self.hop_length)

        self.tempo = tempo
        self.period = 60/tempo
        self.alpha = alpha
        self.time_shift = 0

        if duration is None:
            self.duration = self.len_frames * frame_length / (frame_length/hop_length) / sampling_rate
        else:
            self.duration = duration
        self.part_len_seconds = part_len_seconds
        self.min_delta = min_delta

        self.click_times_sec = get_click_times_sec(self.tempo, self.duration)
        self._find_peaks()


    def _calc_min_number_of_peaks(self) -> int:
        """Calculate minimum number of peaks that should be picked, so we have a higher chance
        of correctly identifying phase."""

        max_number_of_beats = (self.tempo/60) * self.duration  # max possible number of beats in the song

        # it is highly unlikely that all the beats will be picked if we pick only max_number_of_beats peaks
        # that means it could be hard to identify phase, so we should pick more peaks than that. This ratio is
        # defined by the parameter alpha
        min_number_of_peaks = int(max_number_of_beats * self.alpha)
        return min_number_of_peaks

    def _find_peaks(self) -> None:
        """Find peaks in novelty function. """

        min_number_of_peaks = self._calc_min_number_of_peaks()

        self.peaks = peak_pick(self.novelty_function,
                               self.duration,
                               min_number_of_peaks,
                               self.part_len_seconds,
                               self.min_delta)
        self.peak_times = self.frame_times[self.peaks]

    def get_time_shift(self) -> float:
        """Get time shift (beat phase) in seconds."""

        scores = self._try_all_time_shifts()
        max_score = 0
        max_score_shift_ms = 0
        for shift, score in scores.items():
            if score > max_score:
                max_score = score
                max_score_shift_ms = shift

        self.time_shift = max_score_shift_ms / 1000
        return self.time_shift

    def get_beat_track(self) -> np.ndarray:
        time_shift = self.get_time_shift()
        click_times_shifted = self.click_times_sec + time_shift
        return click_times_shifted

    def _get_score(self, shift_ms: int) -> int | float:
        """Calculate score for given time shift.

        :param shift_ms: Time shift in milliseconds.
        :return: Total score calculated for given time shift.
        """

        pass

    def _try_all_time_shifts(self) -> dict:
        """Try all possible time shifts and find the best one.
        Time shifts in the range [0, p] (where p is the beat period) with difference of 1 ms are considered."""

        beat_len = 60 / self.tempo  # length of one beat
        beat_len_ms = beat_len * 1000
        scores = {}
        # used to save the time_shift with max score  + its score
        for i in range(int(beat_len_ms)):
            score = self._get_score(i)
            scores[i] = score

        return scores
