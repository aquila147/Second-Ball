"""Touch detection: sharp changes in ball velocity next to a player."""

import numpy as np
from scipy.signal import find_peaks

MIN_KINK = 4.0      # m/s velocity change; above tracking jitter, below any pass or first touch
MIN_SPACING = 5     # frames between touches (0.2 s)
TOUCH_RADIUS = 2.0  # metres; a player this close can have played the ball


def detect(features, kink=None, min_kink=MIN_KINK, spacing=MIN_SPACING, radius=TOUCH_RADIUS):
    """(frames, teams) of touches. The team is that of the nearest player."""
    kink = np.nan_to_num(features.kink if kink is None else kink)
    frames, _ = find_peaks(kink, height=min_kink, distance=spacing)
    dist = features.dist[frames]
    close = dist.min(axis=1) <= radius
    return frames[close], dist[close].argmin(axis=1)
