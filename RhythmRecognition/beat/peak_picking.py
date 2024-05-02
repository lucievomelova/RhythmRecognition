import numpy as np
import librosa.display


def extract_part_from_novelty(novelty_function: np.ndarray,
                              part_len: int,
                              part_number: int,
                              parts: int,
                              last_part_len: int) -> np.ndarray:

    """Extract only specific part of novelty function and set everything else to 0.

    :param novelty_function: Novelty function of the input signal.
    :param part_len: Length of part in frames.
    :param part_number: Index of current part (-1 for last part).
    :param parts: Total number of parts.
    :param last_part_len: Length of last part.
    :return: Novelty function where everything except for part is set to 0.
    """

    if part_number == -1:  # last part
        part = [0] * (parts * part_len)  # start with 0
        part.extend(novelty_function[-last_part_len:])  # keep last part of novelty function for peak picking
    else:
        part = np.zeros_like(parts*part_len + last_part_len)
        part = [0] * (part_number * part_len)  # fill array with 0 before current part start
        part.extend(novelty_function[part_number * part_len:(part_number + 1) * part_len])  # keep part
        part.extend([0] * ((parts - part_number - 1) * part_len))  # append 0 after part
        part.extend([0] * last_part_len)  # append 0 for last part

    return np.array(part)


def peak_pick_from_part(part: np.ndarray,
                        delta: float,
                        min_number_of_peaks_in_part: int,
                        min_delta: float) -> np.ndarray:
    """Extract peaks from specified part of the novelty function.

    :param part: Part of novelty function
    :param delta: Threshold for mean for peak picking
    :param min_number_of_peaks_in_part: Minimum number of extracted peaks
    :param min_delta: Minimum delta value. After reaching this value for delta, the method will return even if
        it didn't find enough peaks.
    :return: Peaks in part.
    """

    peaks = librosa.util.peak_pick(part, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=delta, wait=5)
    while len(peaks) < min_number_of_peaks_in_part:
        delta /= 1.1
        peaks = librosa.util.peak_pick(part, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=delta, wait=5)
        if delta < min_delta:
            break
    return peaks


def peak_pick(novelty_function: np.ndarray,
              duration: float,
              min_number_of_peaks: int,
              part_len_seconds: int = 10,
              min_delta: float = 0.00001) -> np.ndarray:
    """Extract peaks by dividing the novelty function into parts and finding peaks in each part separately.

    :param novelty_function: Novelty function of the input signal.
    :param duration: Duration in seconds.
    :param min_number_of_peaks: Minimum number of peaks that should be picked.
    :param part_len_seconds: Length of one part in seconds
    :param min_delta: Minimum value of delta for peak picking. After that, the algorithm will end even if it
    doesn't find at least min_number_of_peaks peaks
    :return: Peak times in frames
    """

    parts = int(duration / part_len_seconds)  # how many parts are there
    part_len = int((len(novelty_function)) / parts)  # length of a part in frames
    last_part_len = len(novelty_function) - (parts * part_len)  # length of last part in terms of novelty points
    peaks = []  # here all peaks will be stored

    # find peaks in each part (except for the last part, which will be processed later)
    for i in range(0, parts):
        delta = 100
        part = extract_part_from_novelty(novelty_function, part_len, i, parts, last_part_len)
        peaks_in_part = peak_pick_from_part(part, delta, min_number_of_peaks // parts, min_delta)
        peaks.extend(list(peaks_in_part))

    # last part
    part = extract_part_from_novelty(novelty_function, part_len, -1, parts, last_part_len)

    if last_part_len > 0:
        delta = 100
        min_number_of_peaks = int(min_number_of_peaks / (duration / last_part_len))
        peaks_in_part = peak_pick_from_part(part, delta, min_number_of_peaks, min_delta)
        peaks.extend(list(peaks_in_part))

    return np.array(peaks)
