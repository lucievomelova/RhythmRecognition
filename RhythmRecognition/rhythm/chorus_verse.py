import librosa
import numpy as np

from RhythmRecognition.rhythm.rhythm_recognition import RhythmTracker
from RhythmRecognition.math import rmse, first_order_diff
from RhythmRecognition.constants import *


class ChorusVerseRhythmTracker(RhythmTracker):
    """Rhythm tracking for songs based on dividing songs into chorus and verse parts (and possibly other parts).
    \n
    Song parts are extracted based on root-mean-square energy (RMSE). We will compute RMSE as a segment-level feature
    (in range of seconds). This will give us information about energy of longer parts of the song.
    In theory, verse should have lower energy than chorus, so there should be very clear changes in the energy
    function when going from a verse to a chorus (this should also work for other song part types).
    """

    signal: np.ndarray
    """Audio time series."""

    def __init__(self,
                 audiofile: str,
                 novelty_function: np.ndarray,
                 tempo: int,
                 beat_times: np.ndarray,
                 duration: float | None = None,
                 tolerance_interval: int = 20,
                 alpha: float = 2,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH,
                 segment_length_sec: int = SEGMENT_LENGTH_SECONDS):
        """
        :param audiofile: Name of the audio file to be processed.
        :param novelty_function: Novelty function of the input audio signal.
        :param duration: Duration of the input song in seconds.
        :param tempo: Tempo in BPM.
        :param beat_times: List of beat times.
        :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
            from the novelty function. The base is (tempo/60) * duration.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        :param segment_length_sec: Length of segment for computing energy (in seconds).
        """
        super().__init__(novelty_function, duration, tempo, beat_times, tolerance_interval, alpha,
                         sampling_rate, hop_length, frame_length)
        self.signal, _ = librosa.load(audiofile, duration=duration, sr=self.sampling_rate)  # load the audiofile
        self.segment_size = segment_length_sec*sampling_rate

    def compute_segment_rmse(self) -> np.ndarray:
        """Compute root-mean-square energy of on segment-level."""
        return rmse(self.signal, self.segment_size, self.hop_length)

    def find_song_parts(self) -> np.ndarray:
        """Divide song int verse, chorus and other song parts.

        :return: Array of points where there is a transition from one part to another.
        """
        rmse_segment = self.compute_segment_rmse()
        rmse_diff = first_order_diff(rmse_segment)
        rmse_diff = np.abs(rmse_diff)
        max_parts = int(self.duration // 20)
        part_transitions = [0]
        for i in range(max_parts):
            peak = np.argmax(rmse_diff)
            rmse_diff[peak-(self.sampling_rate*10//self.hop_length):peak+(self.sampling_rate*10//self.hop_length)] = 0
            part_transitions.append(peak)

        part_transitions = librosa.frames_to_time(part_transitions, sr=self.sampling_rate, hop_length=self.hop_length)
        return np.array(part_transitions)

    def find_rhythmic_onsets(self) -> np.ndarray:
        """Extract rhythmic onsets in the song by dividing it into chorus and verse (and possibly other part types)
        parts and extracting note onsets from each aprt separately.

        :return: Rhythmic note onsets.
        """
        parts_times = self.find_song_parts()
        parts_times = np.sort(parts_times)
        picked_onsets = []
        for i in range(len(parts_times)-1):
            start_time_ms = parts_times[i]*1000
            end_time_ms = parts_times[i+1]*1000
            rhythmic_onsets_in_part = self._find_rhythmic_onsets_parts(start_time_ms, end_time_ms)
            picked_onsets.extend(rhythmic_onsets_in_part)

        if parts_times[-1] < self.duration - 5:  # last part
            rhythmic_onsets_in_part = self._find_rhythmic_onsets_parts(parts_times[-1] * 1000, self.duration * 1000)
            picked_onsets.extend(rhythmic_onsets_in_part)

        self.rhythmic_onsets = np.array(picked_onsets)
        return self.rhythmic_onsets
