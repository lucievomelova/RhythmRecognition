import librosa.display
import numpy as np
from RhythmRecognition.constants import *
from RhythmRecognition.beat.peak_picking import peak_pick


class RhythmTracker:
    """Base class for specific rhythm detection approaches.
    \n
    **Rhythm** is the pattern of sounds, silences, and emphases in a song.
    We will focus on strong note onsets that are frequently occurring after beats approximately after
    the same time as other note onsets. To do this, we will calculate score for each distance between
    a closest preceding beat and a note onset candidate. Distances with high scores will be
    considered as frequently occurring. If a note onset candidate is occuring after a beat and this distance
    has a high score, this note onset candidate will be considered to be a rhythmic note onset.
    \n
    Score will be calculated on parts of a song. The specific rhythm tracking approaches differ in
    song partitioning.
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

    beat_times: np.ndarray
    """Beat times in seconds."""

    tolerance: int
    """Length of tolerance interval in milliseconds for scoring."""

    rhythmic_onsets: np.ndarray
    """Note onset times that belong to some rhythmic pattern of the song."""

    sampling_rate: int
    """Defines the number of samples per second taken from a continuous signal to make a discrete signal."""

    frame_length: int
    """Number of samples in a frame."""

    hop_length: int
    """Number of samples by which we have to advance between two consecutive frames."""

    def __init__(self,
                 novelty_function: np.ndarray,
                 duration: float | None,
                 tempo: int,
                 beat_times: np.ndarray,
                 tolerance_interval: int = 10,
                 alpha: float = 2,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param duration: Duration of the input song in seconds.
        :param tempo: Tempo in BPM.
        :param beat_times: List of beat times.
        :param tolerance_interval: Length of tolerance interval in milliseconds.
        :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
            from the novelty function. The base is (tempo/60) * duration.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """

        self.novelty_function = novelty_function
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.beat_times = beat_times
        self.len_frames = len(novelty_function)
        self.frame_times = librosa.frames_to_time(np.arange(len(self.novelty_function)),
                                                  sr=sampling_rate, hop_length=hop_length)

        if duration is None:
            self.duration = self.len_frames * frame_length / (frame_length/hop_length) / sampling_rate
        else:
            self.duration = duration

        self.tolerance = tolerance_interval
        self.tempo = tempo
        self.period = 60/tempo
        self.alpha = alpha
        self._find_peaks()

    def _find_peaks(self) -> None:
        """Find peaks in novelty function. """
        self.peaks = peak_pick(self.novelty_function,
                               self.duration,
                               int((self.tempo/60)*self.duration*self.alpha))
        self.peak_times = self.frame_times[self.peaks]

    def _find_rhythmic_onsets_parts(self, start_time_ms: float, end_time_ms: float) -> list:
        """Find rhythmic note onset in a part of the song specified by start time and end time.

        :param start_time_ms: Start time in milliseconds.
        :param end_time_ms: End time in milliseconds.
        :return: List of extracted rhythmic note onsets.
        """

        shifts_ms = [shift[0] for shift in self._dominant_time_shifts(start_time_ms, end_time_ms)]
        beat_index = 0
        period_ms = self.period * 1000
        rhythmic_peaks_in_part = []

        for peak_time in self.peak_times:
            p = peak_time * 1000  # to milliseconds
            if p < start_time_ms:
                continue
            elif p >= end_time_ms:
                return rhythmic_peaks_in_part

            b = self.beat_times[beat_index] * 1000
            distance = int(p - b)
            while distance > period_ms:
                beat_index += 1
                if beat_index >= len(self.beat_times):
                    break
                b = self.beat_times[beat_index] * 1000
                distance = int(p - b)
            # check if peak with this distance (+- tolerance) was already found
            # add score point to all times in tolerance window
            for j in range(int(distance - self.tolerance), int(distance + self.tolerance)):
                if j in shifts_ms:
                    rhythmic_peaks_in_part.append(peak_time)

        return rhythmic_peaks_in_part

    def _calculate_score(self, start_time_ms: float, end_time_ms: float) -> dict:
        """Calculate score for a part of the song specified by start time and end time.

        :param start_time_ms: Start time in milliseconds.
        :param end_time_ms: End time in milliseconds.
        :return: Score dictionary, where key is time in ms after the closest beat and value is number of note onsets
            corresponding with that time +- tolerance.
        """

        score = {}
        beat_index = 0
        period_ms = self.period * 1000

        for peak_time in self.peak_times:
            p = peak_time * 1000  # to milliseconds
            if p < start_time_ms:
                continue
            elif p >= end_time_ms:
                return dict(sorted(score.items()))

            b = self.beat_times[beat_index] * 1000
            distance = int(p - b)
            while distance > period_ms:
                beat_index += 1
                if beat_index >= len(self.beat_times):
                    break
                b = self.beat_times[beat_index] * 1000
                distance = int(p - b)

            # check if peak with this distance (+- tolerance) was already found
            # add score -1 to all times in tolerance window
            for j in range(int(distance-self.tolerance), int(distance+self.tolerance)):
                j %= int(self.period*1000)
                # if j < 0:  # skip everything smaller than 0 or greater than the whole beat period
                #     continue
                if j in score:
                    score[j] += 1
                else:
                    score[j] = 1
        return dict(sorted(score.items()))

    def _average_score(self, scores: dict) -> dict:
        """Calculate average score of times in the range of the tolerance interval.

        :param scores: Dictionary of scores.
        :return: The given score dictionary, normalized.
        """
        for time, s in scores.items():
            normalization = 1
            for j in range(time - self.tolerance, time + self.tolerance):
                if j < 0 or j > self.duration * 1000:
                    continue
                if j in scores and j != time:
                    scores[time] += scores[j]
                    normalization += 1
            scores[time] /= normalization
        return scores

    def _dominant_time_shifts(self, start_time_ms: float, end_time_ms: float) -> list:
        """Pick note onset time shifts with the highest scores in the given song part.
         Corresponding note onsets should belong to rhythmic note onsets.

        :param start_time_ms: Start of the song part in milliseconds.
        :param end_time_ms: End of the song part in milliseconds.
        :return: List of time shifts with the highest scores.
        """
        score = self._calculate_score(start_time_ms, end_time_ms)
        score = self._average_score(score)
        if score == {}:
            return []
        shifts = []
        for i in range(4):
            shifts.append(max(score.items(), key=lambda k: k[1]))
            time = shifts[i][0]
            for j in range(time - self.tolerance, time + self.tolerance):
                if j < 0 or j > self.duration * 1000:
                    continue
                if j in score:
                    score[j] = 0
        return shifts

    def find_rhythmic_onsets(self) -> np.ndarray:
        """Extract rhythmic onsets in the song by dividing it into chorus and verse (and possibly other part types)
        parts and extracting note onsets from each aprt separately.

        :return: Rhythmic note onsets.
        """
        pass
