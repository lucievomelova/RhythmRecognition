import numpy as np
from RhythmRecognition.onset.novelty_function import NoveltyFunction
from RhythmRecognition.onset.energy import EnergyNovelty
from RhythmRecognition.onset.spectral import SpectralNovelty
from RhythmRecognition.tempo.tempogram import Tempogram
from RhythmRecognition.tempo.fourier import FourierTempogram
from RhythmRecognition.tempo.autocorrelation import AutocorrelationTempogram
from RhythmRecognition.tempo.hybrid import HybridTempogram
from RhythmRecognition.beat.beat_tracker import BeatTracker
from RhythmRecognition.beat.score import ScoreBeatTracker
from RhythmRecognition.beat.penalty import PenaltyBeatTracker
from RhythmRecognition.rhythm.rhythm_tracker import RhythmTracker
from RhythmRecognition.rhythm.parts import EqualPartsRhythmTracker
from RhythmRecognition.rhythm.chorus_verse import ChorusVerseRhythmTracker
from RhythmRecognition.constants import *


"""Module for initializing objects of specific instance for each step of rhythm recognition.
This module is used in detect.py for initializing specific concrete class instance based on given string parameter."""


NOVELTY = ["energy", "spectral"]
TEMPO = ["fourier", "autocorrelation", "hybrid"]
BEAT = ["score", "penalty"]
RHYTHM = ["parts", "chorus-verse"]


class Factory:
    @staticmethod
    def init_novelty(audiofile: str,
                     approach: str,
                     duration: float | None,
                     gamma: int,
                     sampling_rate: int,
                     hop_length: int,
                     frame_length: int) -> NoveltyFunction:

        if approach == "energy":
            novelty = EnergyNovelty(audiofile, duration, gamma,
                                    sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)
        elif approach == "spectral":
            novelty = SpectralNovelty(audiofile, duration, gamma,
                                      sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)
        else:
            raise Exception("Invalid novelty function approach, options are: " + str(NOVELTY))

        return novelty
    
    @staticmethod
    def init_tempogram(approach: str,
                       novelty: np.ndarray,
                       similarity: int = 5,
                       number_of_dominant_values: int = 5,
                       lower_bound: int = 40,
                       upper_bound: int = 200,
                       sampling_rate: int = SAMPLING_RATE,
                       hop_length: int = HOP_LENGTH,
                       frame_length: int = FRAME_LENGTH) -> Tempogram:

        if approach == "fourier":
            tempogram = FourierTempogram(novelty, similarity, number_of_dominant_values, lower_bound, upper_bound,
                                         sampling_rate=sampling_rate,
                                         hop_length=hop_length,
                                         frame_length=frame_length)
        elif approach == "autocorrelation":
            tempogram = AutocorrelationTempogram(novelty, similarity, number_of_dominant_values, lower_bound,
                                                 upper_bound,
                                                 sampling_rate=sampling_rate,
                                                 hop_length=hop_length,
                                                 frame_length=frame_length)
        elif approach == "hybrid":
            tempogram = HybridTempogram(novelty, similarity, number_of_dominant_values, lower_bound, upper_bound,
                                        sampling_rate=sampling_rate, hop_length=hop_length, frame_length=frame_length)
        else:
            raise Exception("Invalid tempogram approach, options are: " + str(TEMPO))
        return tempogram

    @staticmethod
    def init_beat_tracker(approach: str,
                          novelty: np.ndarray,
                          tempo: int,
                          duration: float | None,
                          tolerance_interval: int,
                          alpha: float = 1.5,
                          part_len_seconds: int = 10,
                          min_delta: int = 0.00001,
                          sampling_rate: int = SAMPLING_RATE,
                          hop_length: int = HOP_LENGTH,
                          frame_length: int = FRAME_LENGTH) -> BeatTracker:

        if approach == "score":
            beat_tracker = ScoreBeatTracker(novelty, tempo, duration, tolerance_interval, alpha, part_len_seconds,
                                            min_delta,
                                            sampling_rate=sampling_rate,
                                            hop_length=hop_length,
                                            frame_length=frame_length)
        elif approach == "penalty":
            beat_tracker = PenaltyBeatTracker(novelty, tempo, duration, alpha, part_len_seconds, min_delta,
                                              sampling_rate=sampling_rate,
                                              hop_length=hop_length,
                                              frame_length=frame_length)
        else:
            raise Exception("Invalid beat tracking approach, options are: " + str(BEAT))
        return beat_tracker

    @staticmethod
    def init_rhythm_tracker(approach: str,
                            audiofile: str,
                            novelty: np.ndarray,
                            duration: float | None,
                            bpm: int | None,
                            beats: np.ndarray,
                            tolerance_interval: int = 10,
                            alpha: float = 2,
                            part_len_seconds: int = 20,
                            sampling_rate: int = SAMPLING_RATE,
                            hop_length: int = HOP_LENGTH,
                            frame_length: int = FRAME_LENGTH,
                            segment_length_seconds: int = SEGMENT_LENGTH_SECONDS) -> RhythmTracker:

        if approach == "parts":
            rhythm_tracker = EqualPartsRhythmTracker(novelty, bpm, beats, duration,
                                                     tolerance_interval, alpha, part_len_seconds,
                                                     sampling_rate=sampling_rate, hop_length=hop_length,
                                                     frame_length=frame_length)
        elif approach == "chorus-verse":
            rhythm_tracker = ChorusVerseRhythmTracker(audiofile, novelty, bpm, beats, duration, tolerance_interval,
                                                      alpha,
                                                      sampling_rate=sampling_rate, hop_length=hop_length,
                                                      frame_length=frame_length,
                                                      segment_length_sec=segment_length_seconds)

        else:
            raise Exception("Invalid rhythm tracking approach, options are: " + str(RHYTHM))

        return rhythm_tracker
