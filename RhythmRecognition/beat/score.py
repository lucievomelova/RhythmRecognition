import numpy as np
from RhythmRecognition.beat.beat_tracker import BeatTracker


class ScoreBeatTracker(BeatTracker):
    """Beat tracker based on score.
    \n
    Beat tracking will be based trying out all possible time shifts and assigning score points to shifted time clicks.
    We will assign 1 score point to each shifted time click that is close to a note onset candidate (how close
    a candidate needs to be will be defined by the tolerance interval).
    After trying all possible time shifts, we will choose time shift with the highest score.
    """

    tolerance: int
    """Length of tolerance interval in milliseconds for scoring."""

    def __init__(self,
                 novelty_function: np.ndarray,
                 tempo: int,
                 duration: float | None = None,
                 tolerance_interval: int = 10,
                 alpha: float = 1.5,
                 part_len_seconds: int = 10,
                 min_delta: int = 0.00001):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param tempo: Tempo (in BPM) of the input song.
        :param duration: Duration of the input song in seconds.
        :param tolerance_interval: Length of tolerance interval in milliseconds.
        :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
            from the novelty function. The base is (tempo/60) * duration.
        :param part_len_seconds: Peak  picking is done on smaller parts of the song. This parameter specifies
            the part length in seconds.
        :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
            smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
            ends even before finding the desired number of peaks.
        """
        super().__init__(novelty_function, tempo, duration, alpha, part_len_seconds, min_delta)
        self.tolerance = tolerance_interval

    def _get_score(self, shift_ms: int) -> int:
        """Calculate score for given time shift.

        :param shift_ms: Time shift in milliseconds.
        :return: Total score calculated for given time shift.
        """

        peak_index = 0
        score = 0
        for click_time in self.click_times_sec:
            shifted_click_time_ms = (click_time * 1000) + shift_ms  # click time in ms + time shift in ms
            peak_ms = self.peak_times [peak_index] * 1000  # peak time in ms
            diff = abs(shifted_click_time_ms - peak_ms)  # difference in ms between click time and peak

            # find a peak that is immediately after shifted click time
            while shifted_click_time_ms > peak_ms:
                diff = abs(shifted_click_time_ms - peak_ms)
                peak_index += 1
                if peak_index >= len(self.peak_times):
                    break
                peak_ms = self.peak_times[peak_index] * 1000

            # check which peak is closer to current shifted_click_time_ms - the previous one or the current one?
            if diff > abs(shifted_click_time_ms - peak_ms):
                diff = abs(shifted_click_time_ms - peak_ms)  # it's the current one
            else:
                if peak_index > 0:
                    peak_index -= 1  # go back to the previous one

            # check if this shifted_click_time_ms is in a tolerance window around its closest peak
            if diff <= self.tolerance:
                score += 1  # yes -> increase score

        return score
