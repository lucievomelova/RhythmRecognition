import numpy as np
from RhythmRecognition.constants import *


class Tempogram:
    """Base class for specific tempogram implementations.
    \n
    A **tempogram** indicates specific tempo relevance for each time instance in a song.
    Only an interval of reasonable tempi should be considered for tempo analysis, as too high values
    or too small values do not really make sense in terms of musical tempo analysis.
    \n
    We assume that the input songs have constant tempo and the BPM value is an integer."""

    novelty_function: np.ndarray
    """Novelty function of the input audio signal."""

    similarity: int
    """BPM tolerance that specifies which BPM values belong to the same group."""

    number_of_dominant_values: int
    """How many dominant BPM values should be extracted for later computations."""

    tempogram: np.ndarray | None
    """Tempogram of the input audio file. """

    bpm_values: []
    """List of dominant BPM values."""

    tempo: int | None
    """The most dominant BPM value that should indicate tempo of the song."""

    lower_bound: int
    """Lowest possible BPM value that will be considered in tempo analysis."""

    upper_bound: int
    """Highest possible BPM value that will be considered in tempo analysis."""

    sampling_rate: int
    """Defines the number of samples per second taken from a continuous signal to make a discrete signal."""

    frame_length: int
    """Number of samples in a frame."""

    hop_length: int
    """Number of samples by which we have to advance between two consecutive frames."""

    def __init__(self, novelty_function: np.ndarray,
                 similarity: int = 5,
                 number_of_dominant_values: int = 5,
                 lower_bound: int = 40,
                 upper_bound: int = 200,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH):
        """
        :param novelty_function: Novelty function of the input audio signal.
        :param similarity: BPM tolerance that specifies which BPM values belong to the same group.
        :param number_of_dominant_values: How many dominant BPM values should be extracted for later computations.
        :param lower_bound: Lowest possible BPM value that will be considered.
        :param upper_bound: Highest possible BPM value that will be considered.
        :param sampling_rate: Defines the number of samples per second taken from a continuous signal
         to make a discrete signal.
        :param frame_length: Number of samples in a frame
        :param hop_length: Number of samples by which we have to advance between two consecutive frames.
        """
        self.novelty_function = novelty_function
        self.similarity = similarity
        self.number_of_dominant_values = number_of_dominant_values
        self.tempogram = None
        self.bpm_values = []
        self.tempo = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _group_similar_bpms(self, bpm_list: []) -> dict:
        """Group similar BPM values together, because there is a high chance that they represent the same
        dominant BPM value.

        :param bpm_list: list of extracted dominant bpm values
        :return: list of groups (each group is also a list and contains similar BPMs)"""
        groups = {}
        for bpm in bpm_list:
            appended = False
            for key in groups.keys():
                if abs(bpm - key) <= self.similarity:  # if a similar group already exists, add this bpm to it
                    groups[key].append(bpm)
                    appended = True
            if not appended:  # otherwise create a new group for it
                groups[bpm] = [bpm]
        return groups

    @staticmethod
    def _get_dominant_group_avg(groups, weights) -> int:
        """Get the average value of the dominant BPM group.

        :param groups: BPM groups
        :param weights: weght that each bpm has (based on how dominant its values was in the tempogram)
        :return: the computed most dominant BPM
        """
        dominant_bpm = None
        dominant_weight = 0
        for group in groups.values():
            bpm_sum = 0  # total sum of numbers of the group
            weight = 0  # total weight of group
            for bpm in group:
                bpm_sum += bpm * weights[bpm]
                weight += weights[bpm]
            average = round(bpm_sum / weight)
            if dominant_weight < weight:
                dominant_weight = weight
                dominant_bpm = average
        return dominant_bpm

    def _find_dominant_BPM(self, bpm_list: []) -> int:
        """Find the most probable tempo of the song from the list of dominant BPM values.

        :param bpm_list: dominant BPM values
        :return: computed song BPM
        """
        weights = {}
        for i in range(len(bpm_list)):
            weights[bpm_list[i]] = len(bpm_list) - i  # the closer to the start of the bpm_list, the higher the weight

        # group similar BPMs together:
        similar_bpm_groups = self._group_similar_bpms(bpm_list)

        # find the group with the highest weight and take it's rounded average as the final dominant bpm
        dominant_bpm = self._get_dominant_group_avg(similar_bpm_groups, weights)

        return dominant_bpm

    def _get_possible_BPM_from_tempogram(self) -> list:
        """Analyze the fourier tempogram and extract bpm values that are dominant.
        :return: List of dominant BPM values (its size will be at least self.number_of_dominant_values)
        """
        # get list of BPMs that the fourier tempogram represents
        max_bpms = []  # this list will hold BPMs with very high values - possible BPMs of the song
        sum_of_tempos = np.sum(self.tempogram.T, axis=0)  # sum of values for all tempos in tempogram

        while len(max_bpms) < self.number_of_dominant_values:
            max_bpm_indices = []
            for i in range(self.number_of_dominant_values):
                max_bpm = np.argmax(sum_of_tempos)
                max_bpm_indices.append(max_bpm)
                sum_of_tempos[max_bpm] = 0
            max_bpms.extend([i for i in self.bpm_values[max_bpm_indices] if self.lower_bound <= i <= self.upper_bound])

        return max_bpms

    def _compute_tempogram(self) -> None:
        """Compute tempogram for given novelty function."""
        pass

    def _analyze_tempo(self) -> None:
        """Analyze the computed tempogram to find dominant tempo."""
        if self.tempogram is None:
            self._compute_tempogram()

    def get_tempogram(self) -> np.ndarray:
        """Get the computed tempogram."""
        self._compute_tempogram()
        return self.tempogram

    def get_tempo(self) -> int:
        """Get the most dominant tempo."""
        self._compute_tempogram()
        self._analyze_tempo()

        return self.tempo
