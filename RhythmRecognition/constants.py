"""Constants used for audio file processing."""


FRAME_LENGTH = 2048
"""**Frame length** specifies the number of samples in a frame.
The input signal is cut into smaller frames (of this size) and calculations are done over them."""

HOP_LENGTH = 512
"""**Hop length** refers to the number of samples by which we have to advance between two consecutive frames. 
In other words, it determines the overlap between frames."""

SAMPLING_RATE = 44100
"""**Sampling rate** specifies the number of samples per second that were take from a continuous signal to 
make a discrete signal. This sampling rate (so 44.1 kHz) is the standard sampling rate for audio formats."""

SEGMENT_LENGTH_SECONDS = 8
"""Length of segment in seconds for calculating segment-level features."""


SEGMENT_LENGTH = SAMPLING_RATE * SEGMENT_LENGTH_SECONDS
"""Length of segment in frames for calculating segment-level features."""

WIN_LENGTH = 1024
"""Window length used in autocorrelation tempogram."""
