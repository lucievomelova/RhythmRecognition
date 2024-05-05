import numpy as np
import librosa
from RhythmRecognition.onset.energy import EnergyNovelty
from RhythmRecognition.onset.spectral import SpectralNovelty
from RhythmRecognition.tempo.fourier import FourierTempogram
from RhythmRecognition.tempo.autocorrelation import AutocorrelationTempogram
from RhythmRecognition.tempo.hybrid import HybridTempogram
from RhythmRecognition.tempo.tempogram import Tempogram
from RhythmRecognition.beat.score import ScoreBeatTracker
from RhythmRecognition.beat.beat_tracker import BeatTracker
from RhythmRecognition.beat.penalty import PenaltyBeatTracker
from RhythmRecognition.rhythm.parts import EqualPartsRhythmTracker
from RhythmRecognition.rhythm.chorus_verse import ChorusVerseRhythmTracker


NOVELTY = ["energy", "spectral"]
TEMPO = ["fourier", "autocorrelation", "hybrid"]
BEAT = ["score", "penalty"]
RHYTHM = ["parts", "chorus-verse"]


def __init_tempogram(approach: str,
                     novelty: np.ndarray,
                     similarity: int = 5,
                     number_of_dominant_values: int = 5,
                     lower_bound: int = 40,
                     upper_bound: int = 200) -> Tempogram:
    if approach == "fourier":
        tempogram = FourierTempogram(novelty, similarity, number_of_dominant_values, lower_bound, upper_bound)
    elif approach == "autocorrelation":
        tempogram = AutocorrelationTempogram(novelty, similarity, number_of_dominant_values, lower_bound, upper_bound)
    elif approach == "hybrid":
        tempogram = HybridTempogram(novelty, similarity, number_of_dominant_values, lower_bound, upper_bound)
    else:
        raise Exception("Invalid tempogram approach, options are: " + str(TEMPO))
    return tempogram


def __init_beat_tracker(approach: str,
                        novelty: np.ndarray,
                        tempo: int,
                        duration: float,
                        tolerance_interval: int,
                        alpha: float = 1.5,
                        part_len_seconds: int = 10,
                        min_delta: int = 0.00001) -> BeatTracker:
    if approach == "score":
        beat_tracker = ScoreBeatTracker(novelty, tempo, duration, tolerance_interval, alpha, part_len_seconds, min_delta)
    elif approach == "penalty":
        beat_tracker = PenaltyBeatTracker(novelty, tempo, duration, alpha, part_len_seconds, min_delta)
    else:
        raise Exception("Invalid beat tracking approach, options are: " + str(BEAT))
    return beat_tracker


def novelty_function(audiofile: str,
                     approach: str = "spectral",
                     duration: float | None = None,
                     gamma: int = 10) -> np.ndarray:
    """Compute novelty function of the given song.

    :param audiofile: Name of the input audio file.
    :param approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
    :param gamma: Compression factor for logarithmic compression.
    :return: Computed novelty function.
    """
    if approach == "energy":
        novelty = EnergyNovelty(audiofile, duration, gamma)
    elif approach == "spectral":
        novelty = SpectralNovelty(audiofile, duration, gamma)
    else:
        raise Exception("Invalid novelty function approach, options are: " + str(NOVELTY))

    result = novelty.get()
    return result


def tempo(audiofile: str,
          approach: str = "fourier",
          novelty_approach: str = "spectral",
          duration: float | None = None,
          novelty_gamma: int = 10,
          similarity: int = 5,
          number_of_dominant_values: int = 5,
          lower_bound: int = 40,
          upper_bound: int = 200) -> int:
    """Get dominant tempo of the given song.

    :param audiofile: Name of the input audio file.
    :param approach: Which tempogram to use - options are fourier, autocorrelation or hybrid.
    :param novelty_approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
    :param novelty_gamma: Compression factor for logarithmic compression used for computing the novelty function.
    :param similarity: BPM tolerance that specifies which BPM values belong to the same group.
    :param number_of_dominant_values: How many dominant BPM values should be extracted for later computations.
    :param lower_bound: Lowest possible BPM value that will be considered.
    :param upper_bound: Highest possible BPM value that will be considered.
    :return: Most dominant tempo value (in BPM).
    """
    novelty = novelty_function(audiofile, novelty_approach, duration, novelty_gamma)
    t = __init_tempogram(approach, novelty, similarity, number_of_dominant_values, lower_bound, upper_bound)
    result = t.get_tempo()
    return result


def tempogram(audiofile: str, approach: str = "fourier",
              novelty_approach: str = "spectral",
              duration: float | None = None,
              novelty_gamma: int = 10,
              similarity: int = 5,
              number_of_dominant_values: int = 5,
              lower_bound: int = 40,
              upper_bound: int = 200) -> np.ndarray:
    """Get tempogram of the given song.

    :param audiofile: Name of the input audio file.
    :param approach: Which tempogram to use - options are fourier, autocorrelation or hybrid.
    :param novelty_approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
    :param novelty_gamma: Compression factor for logarithmic compression used for computing the novelty function.
    :param similarity: BPM tolerance that specifies which BPM values belong to the same group.
    :param number_of_dominant_values: How many dominant BPM values should be extracted for later computations.
    :param lower_bound: Lowest possible BPM value that will be considered.
    :param upper_bound: Highest possible BPM value that will be considered.
    :return: Computed tempogram.
    """

    novelty = novelty_function(audiofile, novelty_approach, duration, novelty_gamma)
    t = __init_tempogram(approach, novelty, similarity, number_of_dominant_values, lower_bound, upper_bound)
    result = t.get_tempogram()
    return result


def beat_track(audiofile: str,
               approach: str = "score",
               tempo_approach: str = "fourier",
               novelty_approach: str = "spectral",
               duration: float | None = None,
               novelty_gamma: int = 10,
               bpm: int | None = None,
               similarity_tempo: int = 5,
               number_of_dominant_values_tempo: int = 5,
               lower_bound: int = 40,
               upper_bound: int = 200,
               tolerance_interval: int = 10,
               alpha: float = 1.5,
               part_len_seconds: int = 10,
               min_delta: int = 0.00001) -> np.ndarray:
    """Compute beat track for the given song.

    :param audiofile: Name of the input audio file.
    :param approach: Which beat tracking approach to use - options are score or penalty.
    :param tempo_approach: Which tempogram to use - options are fourier, autocorrelation or hybrid.
    :param novelty_approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
    :param novelty_gamma: Compression factor for logarithmic compression used for computing the novelty function.
    :param bpm: Tempo of the song. Optional parameter - if it is not specified, tempo will be calculated using
        the specified tempo analysis approach. If this paramter is specified, all tempo analysis-related parameters
        will be ignored.
    :param similarity_tempo: BPM tolerance that specifies which BPM values belong to the same group.
    :param number_of_dominant_values_tempo: How many dominant BPM values should be extracted for later computations.
    :param lower_bound: Lowest possible BPM value that will be considered.
    :param upper_bound: Highest possible BPM value that will be considered.
    :param tolerance_interval: Length of tolerance interval in milliseconds.
     Only used in "score" beat tracking approach.
    :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
        from the novelty function. The base is (tempo/60) * duration.
    :param part_len_seconds: Peak  picking is done on smaller parts of the song. This parameter specifies
        the part length in seconds.
    :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
        smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
        ends even before finding the desired number of peaks.
    :return: Array of beat times.
    """

    # set duration to whole song duration if not specified
    if duration is None:
        duration = librosa.get_duration(path=audiofile)
    else:
        duration = librosa.get_duration(path=audiofile)

    novelty = novelty_function(audiofile, novelty_approach, duration, novelty_gamma)
    if bpm is None:
        bpm = tempo(audiofile, tempo_approach, novelty_approach, duration, novelty_gamma,
                    similarity_tempo, number_of_dominant_values_tempo, lower_bound, upper_bound)
    beat_tracker = __init_beat_tracker(approach, novelty, bpm, duration, tolerance_interval, alpha,
                                       part_len_seconds, min_delta)
    result = beat_tracker.get_beat_track()
    return result


def beat_time_shift(audiofile: str,
                    approach: str = "score",
                    tempo_approach: str = "fourier",
                    novelty_approach: str = "spectral",
                    duration: float | None = None,
                    novelty_gamma: int = 10,
                    bpm: int | None = None,
                    similarity_tempo: int = 5,
                    number_of_dominant_values_tempo: int = 5,
                    lower_bound: int = 40,
                    upper_bound: int = 200,
                    tolerance_interval: int = 10,
                    alpha: float = 1.5,
                    part_len_seconds: int = 10,
                    min_delta: int = 0.00001) -> float:
    """Compute beat time shift for the given song. The time shift specifies the time by which we need to shift a click
    track set to a found tempo so that it's clicks will align with the song beats.

    :param audiofile: Name of the input audio file.
    :param approach: Which beat tracking approach to use - options are score or penalty.
    :param tempo_approach: Which tempogram to use - options are fourier, autocorrelation or hybrid.
    :param novelty_approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
        part of the song. If not specified, it is set to the whole song duration.
    :param novelty_gamma: Compression factor for logarithmic compression used for computing the novelty function.
    :param bpm: Tempo of the song. Optional parameter - if it is not specified, tempo will be calculated using
        the specified tempo analysis approach. If this paramter is specified, all tempo analysis-related parameters
        will be ignored.
    :param similarity_tempo: BPM tolerance that specifies which BPM values belong to the same group.
    :param number_of_dominant_values_tempo: How many dominant BPM values should be extracted for later computations.
    :param lower_bound: Lowest possible BPM value that will be considered.
    :param upper_bound: Highest possible BPM value that will be considered.
    :param tolerance_interval: Length of tolerance interval in milliseconds.
     Only used in "score" beat tracking approach.
    :param alpha: Parameter for peak picking specifying the ratio for how many peaks should be extracted
        from the novelty function. The base is (tempo/60) * duration.
    :param part_len_seconds: Peak  picking is done on smaller parts of the song. This parameter specifies
        the part length in seconds.
    :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
        smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
        ends even before finding the desired number of peaks.
    :return: Beat time shift.
    """

    # set duration to whole song duration if not specified
    if duration is None:
        duration = librosa.get_duration(path=audiofile)
    else:
        duration = librosa.get_duration(path=audiofile)

    novelty = novelty_function(audiofile, novelty_approach, duration, novelty_gamma)
    if bpm is None:
        bpm = tempo(audiofile, tempo_approach, novelty_approach, duration, novelty_gamma,
                    similarity_tempo, number_of_dominant_values_tempo, lower_bound, upper_bound)
    beat_tracker = __init_beat_tracker(approach, novelty, bpm, duration, tolerance_interval, alpha,
                                       part_len_seconds, min_delta)
    result = beat_tracker.get_time_shift()
    return result


def rhythm_track(audiofile: str,
                 approach: str = "parts",
                 beat_approach: str = "score",
                 tempo_approach: str = "fourier",
                 novelty_approach: str = "spectral",
                 duration: float | None = None,
                 novelty_gamma: int = 10,
                 bpm: int | None = None,
                 similarity_tempo: int = 5,
                 number_of_dominant_values_tempo: int = 5,
                 lower_bound: int = 40,
                 upper_bound: int = 200,
                 tolerance_beat: int = 10,
                 alpha_beat: float = 1.5,
                 part_len_seconds_beat: int = 10,
                 min_delta: int = 0.00001,
                 tolerance_interval: int = 10,
                 alpha: float = 2,
                 part_len_seconds: int = 20) -> np.ndarray:
    """Find rhythmic notes.

    :param audiofile: Name of the input audio file.
    :param approach: Which rhythm tracking approach to use - options are score or chorus-verse. Use chorus-verse only
        if the song has significant energy changes between different song parts.
    :param beat_approach: Which beat tracking approach to use - options are score or penalty.
    :param tempo_approach: Which tempogram to use - options are fourier, autocorrelation or hybrid.
    :param novelty_approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
        part of the song. If not specified, it is set to the whole song duration.
    :param novelty_gamma: Compression factor for logarithmic compression used for computing the novelty function.
    :param bpm: Tempo of the song. Optional parameter - if it is not specified, tempo will be calculated using
        the specified tempo analysis approach. If this paramter is specified, all tempo analysis-related parameters
        will be ignored.
    :param similarity_tempo: BPM tolerance that specifies which BPM values belong to the same group.
    :param number_of_dominant_values_tempo: How many dominant BPM values should be extracted for later computations.
    :param lower_bound: Lowest possible BPM value that will be considered.
    :param upper_bound: Highest possible BPM value that will be considered.
    :param tolerance_beat: Length of tolerance interval in milliseconds for beat tracking.
        Only used in "score" beat tracking approach.
    :param alpha_beat: Parameter for peak picking sued in beat tracking specifying the ratio for how many
        peaks should be extracted from the novelty function. The base is (tempo/60) * duration.
    :param part_len_seconds_beat: Peak  picking is done on smaller parts of the song. This parameter specifies
        the part length in seconds.
    :param min_delta: Delta specifies the threshold for peak picking. Peak picking algorithm slowly makes delta
        smaller so that the correct number of peaks is extracted. When delta reaches min_delta, the algorithm
        ends even before finding the desired number of peaks.
    :param tolerance_interval: Length of tolerance interval in milliseconds.
    :param alpha: Parameter for peak picking used in rhythm tracking specifying the ratio for how many peaks
        should be extracted from the novelty function. The base is (tempo/60) * duration.

    :param part_len_seconds: Length of song part in seconds. Peak picking will be done on smaller parts
        of the song of the specified length. Only used in "parts" rhythm tracking approach.
    :return: Array of rhythmic note onset times.
    """

    # set duration to whole song duration if not specified
    if duration is None:
        duration = librosa.get_duration(path=audiofile)
    else:
        duration = librosa.get_duration(path=audiofile)

    novelty = novelty_function(audiofile, novelty_approach, duration, novelty_gamma)
    if bpm is None:
        bpm = tempo(audiofile, tempo_approach, novelty_approach, duration, novelty_gamma,
                  similarity_tempo, number_of_dominant_values_tempo, lower_bound, upper_bound)
    beats = beat_track(audiofile, beat_approach, tempo_approach, novelty_approach,
                       duration, novelty_gamma,
                       similarity_tempo, number_of_dominant_values_tempo, lower_bound, upper_bound,
                       tolerance_beat, alpha_beat, part_len_seconds_beat, min_delta)

    if approach == "parts":
        rhythm_tracker = EqualPartsRhythmTracker(novelty, duration, bpm, beats,
                                                 tolerance_interval, alpha, part_len_seconds)
    elif approach == "chorus-verse":
        rhythm_tracker = ChorusVerseRhythmTracker(audiofile, novelty, duration, bpm, beats, tolerance_interval, alpha)

    else:
        raise Exception("Invalid rhythm tracking approach, options are: " + str(RHYTHM))
    result = rhythm_tracker.find_rhythmic_onsets()
    return result
