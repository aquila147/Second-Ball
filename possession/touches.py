"""Touch detection: sharp changes in ball velocity next to a player."""

import numpy as np
from scipy.signal import find_peaks

MIN_KINK = 4.0      # m/s velocity change; above tracking jitter, below any pass or first touch
MIN_SPACING = 5     # frames between touches (0.2 s)
TOUCH_RADIUS = 2.0  # metres; a player this close can have played the ball


def detect(features):
    """(frames, teams) of touches. The team is that of the nearest player."""
    kink = np.nan_to_num(features.kink)
    frames, _ = find_peaks(kink, height=MIN_KINK, distance=MIN_SPACING)
    dist = features.dist[frames]
    close = dist.min(axis=1) <= TOUCH_RADIUS
    return frames[close], dist[close].argmin(axis=1)
