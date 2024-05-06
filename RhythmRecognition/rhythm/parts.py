import numpy as np
from RhythmRecognition.rhythm.rhythm_tracker import RhythmTracker
from RhythmRecognition.constants import *


class EqualPartsRhythmTracker(RhythmTracker):
    """Rhythm tracking for songs based on dividing songs into equal parts.
    \n
    If verse and chorus detection cannot be applied, because the song energy changes are not giving any information
    about part transitions, we will simply divide the song into equal parts and find rhythmic note onsets in each
    part separately.
    """

    part_len_seconds: int
    """Length of part in seconds. Peak picking will be done on smaller parts of the song of the specified length."""

    def __init__(self,
                 novelty_function: np.ndarray,
                 tempo: int,
                 beat_times: np.ndarray,
                 duration: float | None = None,
                 tolerance_interval: int = 20,
                 alpha: float = 2,
                 num_rhythmic_onsets: int = 4,
                 part_len: int = 20,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param duration: Duration of the input song in seconds.
        :param tempo: Tempo in BPM.
        :param beat_times: List of beat times.
        :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
            from the novelty function. The base is (tempo/60) * duration.
        :param part_len: Length of song part in seconds. Peak picking will be done on smaller parts
            of the song of the specified length.
        :param num_rhythmic_onsets: Number that specifies how many groups of note onsets with high score should be
         put in the rhythm track.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """
        super().__init__(novelty_function, duration, tempo, beat_times, tolerance_interval, alpha, num_rhythmic_onsets,
                         sampling_rate, hop_length, frame_length)

        self.part_len_seconds = part_len

    def find_rhythmic_onsets(self) -> np.ndarray:
        """Extract rhythmic onsets in the song by dividing it into equal parts and extracting
        onsets from each part separately.

        :param part_len: Length of song part
        :return: Rhythmic note onsets.
        """
        picked_onsets = []
        for i in range(int(self.duration // self.part_len_seconds)):
            rhythmic_onsets_in_part = self._find_rhythmic_onsets_parts((i * self.part_len_seconds) * 1000,
                                                                       ((i+1)*self.part_len_seconds) * 1000)
            picked_onsets.extend(rhythmic_onsets_in_part)

        # last part
        i = self.duration // self.part_len_seconds
        rhythmic_onsets_in_part = self._find_rhythmic_onsets_parts((i * self.part_len_seconds) * 1000,
                                                                   self.duration * 1000)
        picked_onsets.extend(rhythmic_onsets_in_part)

        self.rhythmic_onsets = np.array(picked_onsets)
        return self.rhythmic_onsets
