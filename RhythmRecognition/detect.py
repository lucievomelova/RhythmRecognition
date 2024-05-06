from RhythmRecognition.rhythm_recognition_factory import *

"""Module with wrapper methods that take audiofile and specific approaches in each step as input and do all of 
the necessary computations so that the user doesn't have to compute each step separately."""


def novelty_function(audiofile: str,
                     approach: str = "spectral",
                     duration: float | None = None,
                     gamma: int = 10,
                     sampling_rate: int = SAMPLING_RATE,
                     hop_length: int = HOP_LENGTH,
                     frame_length: int = FRAME_LENGTH) -> np.ndarray:
    """Compute novelty function of the given song.

    :param audiofile: Name of the input audio file.
    :param approach: Which novelty function to compute - options are spectral or energy.
    :param duration: Duration of the song in seconds. Only specify this parameter if you need to use a smaller
            part of the song. If not specified, it is set to the whole song duration.
    :param gamma: Compression factor for logarithmic compression.
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :return: Computed novelty function.
    """
    novelty = Factory.init_novelty(audiofile=audiofile,
                                   approach=approach,
                                   duration=duration,
                                   gamma=gamma,
                                   sampling_rate=sampling_rate,
                                   hop_length=hop_length,
                                   frame_length=frame_length)
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
          upper_bound: int = 200,
          sampling_rate: int = SAMPLING_RATE,
          hop_length: int = HOP_LENGTH,
          frame_length: int = FRAME_LENGTH) -> int:
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
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :return: Most dominant tempo value (in BPM).
    """
    novelty = novelty_function(audiofile=audiofile, approach=novelty_approach, duration=duration, gamma=novelty_gamma,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    t = Factory.init_tempogram(approach=approach, novelty=novelty, similarity=similarity,
                               number_of_dominant_values=number_of_dominant_values,
                               lower_bound=lower_bound, upper_bound=upper_bound,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)
    result = t.get_tempo()
    return result


def tempogram(audiofile: str, approach: str = "fourier",
              novelty_approach: str = "spectral",
              duration: float | None = None,
              novelty_gamma: int = 10,
              similarity: int = 5,
              number_of_dominant_values: int = 5,
              lower_bound: int = 40,
              upper_bound: int = 200,
              sampling_rate: int = SAMPLING_RATE,
              hop_length: int = HOP_LENGTH,
              frame_length: int = FRAME_LENGTH) -> np.ndarray:
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
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :return: Computed tempogram.
    """

    novelty = novelty_function(audiofile=audiofile, approach=novelty_approach, duration=duration, gamma=novelty_gamma,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    t = Factory.init_tempogram(approach=approach, novelty=novelty, similarity=similarity,
                               number_of_dominant_values=number_of_dominant_values,
                               lower_bound=lower_bound, upper_bound=upper_bound,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

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
               min_delta: int = 0.00001,
               sampling_rate: int = SAMPLING_RATE,
               hop_length: int = HOP_LENGTH,
               frame_length: int = FRAME_LENGTH) -> np.ndarray:
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
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :return: Array of beat times.
    """

    novelty = novelty_function(audiofile=audiofile, approach=novelty_approach, duration=duration, gamma=novelty_gamma,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    if bpm is None:
        bpm = tempo(audiofile=audiofile, approach=tempo_approach,
                    novelty_approach=novelty_approach, duration=duration, novelty_gamma=novelty_gamma,
                    similarity=similarity_tempo, number_of_dominant_values=number_of_dominant_values_tempo,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    beat_tracker = Factory.init_beat_tracker(approach, novelty, bpm, duration, tolerance_interval, alpha,
                                             part_len_seconds, min_delta,
                                             sampling_rate=sampling_rate, hop_length=hop_length,
                                             frame_length=frame_length)
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
                    min_delta: int = 0.00001,
                    sampling_rate: int = SAMPLING_RATE,
                    hop_length: int = HOP_LENGTH,
                    frame_length: int = FRAME_LENGTH) -> float:
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
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :return: Beat time shift.
    """

    novelty = novelty_function(audiofile=audiofile, approach=novelty_approach, duration=duration, gamma=novelty_gamma,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    if bpm is None:
        bpm = tempo(audiofile=audiofile, approach=tempo_approach,
                    novelty_approach=novelty_approach, duration=duration, novelty_gamma=novelty_gamma,
                    similarity=similarity_tempo, number_of_dominant_values=number_of_dominant_values_tempo,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    beat_tracker = Factory.init_beat_tracker(approach=approach, novelty=novelty, tempo=bpm, duration=duration,
                                             tolerance_interval=tolerance_interval, alpha=alpha,
                                             part_len_seconds=part_len_seconds, min_delta=min_delta,
                                             sampling_rate=sampling_rate, hop_length=hop_length,
                                             frame_length=frame_length)
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
                 part_len_seconds: int = 20,
                 sampling_rate: int = SAMPLING_RATE,
                 hop_length: int = HOP_LENGTH,
                 frame_length: int = FRAME_LENGTH,
                 segment_length_seconds: int = SEGMENT_LENGTH_SECONDS) -> np.ndarray:
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
    :param sampling_rate: Defines the number of samples per second taken from a continuous signal
     to make a discrete signal.
    :param frame_length: Number of samples in a frame.
    :param hop_length: Number of samples by which we have to advance between two consecutive frames.
    :param segment_length_seconds: Length of segment for computing energy (in seconds).
    :return: Array of rhythmic note onset times.
    """

    novelty = novelty_function(audiofile=audiofile, approach=novelty_approach, duration=duration, gamma=novelty_gamma,
                               sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    if bpm is None:
        bpm = tempo(audiofile=audiofile, approach=tempo_approach,
                    novelty_approach=novelty_approach, duration=duration, novelty_gamma=novelty_gamma,
                    similarity=similarity_tempo, number_of_dominant_values=number_of_dominant_values_tempo,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    beats = beat_track(audiofile=audiofile, approach=beat_approach, tempo_approach=tempo_approach,
                       novelty_approach=novelty_approach, duration=duration, novelty_gamma=novelty_gamma,
                       bpm=bpm, similarity_tempo=similarity_tempo,
                       number_of_dominant_values_tempo=number_of_dominant_values_tempo,
                       lower_bound=lower_bound, upper_bound=upper_bound,
                       tolerance_interval=tolerance_beat, alpha=alpha_beat, part_len_seconds=part_len_seconds_beat,
                       min_delta=min_delta,
                       sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)

    rhythm_tracker = Factory.init_rhythm_tracker(approach=approach,
                                                 audiofile=audiofile,
                                                 novelty=novelty,
                                                 duration=duration,
                                                 bpm=bpm,
                                                 beats=beats,
                                                 tolerance_interval=tolerance_interval,
                                                 alpha=alpha,
                                                 part_len_seconds=part_len_seconds,
                                                 sampling_rate=sampling_rate,
                                                 hop_length=hop_length,
                                                 frame_length=frame_length,
                                                 segment_length_seconds=segment_length_seconds)

    result = rhythm_tracker.find_rhythmic_onsets()
    return result
