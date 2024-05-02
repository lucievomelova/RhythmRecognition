import numpy as np
import librosa.display
from RhythmRecognition.constants import *

"""Click track generator."""


def shift_click_track(len_frames: int, click_track: np.ndarray, time_shift: float) -> np.ndarray:
    """Shift click track by specified time shift."""

    # Pad the click track with zeros to align with the song
    padded_click = np.pad(click_track, (int(time_shift * SAMPLING_RATE), 0))

    # Ensure both signals have the same length
    min_length = min(len_frames, len(padded_click))
    padded_click = padded_click[:min_length]
    return padded_click


def get_click_times_sec(tempo, duration) -> np.ndarray:
    """Get click times in seconds

    :param tempo: Tempo in BPM.
    :param duration: Duration in seconds.
    :return: Click times (in seconds).
    """
    interval = 60 / tempo
    curr_time = 0
    click_times_arr = [curr_time]
    while curr_time < duration:
        curr_time += interval
        click_times_arr.append(curr_time)

    return np.array(click_times_arr)


def get_click_times_frames(tempo, duration) -> np.ndarray | int:
    """Get click times in frames.

    :param tempo: Tempo in BPM.
    :param duration: Duration in seconds.
    :return: Click times (in frames).
    """
    click_times_sec = get_click_times_sec(tempo, duration)
    frames = librosa.time_to_frames(click_times_sec, sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
    return frames
