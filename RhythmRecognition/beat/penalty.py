import numpy as np
from RhythmRecognition.beat.beat_tracker import BeatTracker
from RhythmRecognition.constants import *


class PenaltyBeatTracker(BeatTracker):
    """Beat tracker based on scores with penalty.
     \n
     Beat tracking will be based trying out all possible time shifts and assigning score points to shifted time clicks.
     But instead of simply awarding 0 or 1 score point, we will award a number of points based on how far the closest
     note onset candidate is from current time click. The closer the candidate is, the more
     points it gets. To do that, we will use a penalty function.
     After trying all possible time shifts, we will choose time shift with the highest score.
    """

    def __init__(self,
                 novelty_function: np.ndarray,
                 tempo: int,
                 duration: float | None = None,
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
        :param part_len_seconds: Peak  picking is done on smaller parts of the song. This parameter specifies
            the part length in seconds.
        :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
            smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
            ends even before finding the desired number of peaks.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """

        super().__init__(novelty_function, tempo, duration, alpha, part_len_seconds, min_delta,
                         sampling_rate, hop_length, frame_length)

    def _penalty(self, distance: float) -> float:
        """Compute penalty for given distance.

        :param distance: Distance to the closest note onset candidate.
        :return: Penalty in range of [0, 1]
        """
        if distance >= self.period/2:
            return 1  # max penalty

        return distance**2 * (4 / (self.period**2))  # normalized quadratic penalty

    def _get_score(self, shift_ms: int) -> float:
        """Calculate score for given time shift.

        :param shift_ms: Time shift in milliseconds.
        :return: Total score calculated for given time shift.
        """

        peak_index = 0
        score = 0
        for click_time in self.click_times_sec:
            shifted_click_time_ms = (click_time * 1000) + shift_ms  # click time in ms + time shift in ms
            peak_ms = self.peak_times [peak_index] * 1000  # peak time in ms
            distance = abs(shifted_click_time_ms - peak_ms)  # distance in ms between click time and peak

            # find a peak that is immediately after shifted click time
            while shifted_click_time_ms > peak_ms:
                distance = abs(shifted_click_time_ms - peak_ms)
                peak_index += 1
                if peak_index >= len(self.peak_times):
                    break
                peak_ms = self.peak_times[peak_index] * 1000

            # check which peak is closer to current shifted_click_time_ms - the previous one or the current one?
            if distance > abs(shifted_click_time_ms - peak_ms):
                distance = abs(shifted_click_time_ms - peak_ms)  # it's the current one
            else:
                if peak_index > 0:
                    peak_index -= 1  # go back to the previous one

            # calculate penalty and add to total score
            penalty = self._penalty(distance)
            score += 1 - penalty

        return score
