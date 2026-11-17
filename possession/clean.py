"""Ball gap filling, smoothing and attack direction."""

import numpy as np
from scipy.signal import savgol_filter

from .io import AWAY, FPS, HOME

MAX_GAP = 10        # frames (0.4 s): short enough for a straight line to be a fair guess
SMOOTH_WINDOW = 7   # frames
SMOOTH_ORDER = 2
KINK_SPAN = 3       # frames either side used to measure a velocity change


def runs(mask):
    """Half-open (start, stop) spans where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_gaps(ball):
    """Ball with gaps of up to MAX_GAP frames interpolated, and the mask of usable frames."""
    seen = ~np.isnan(ball[:, 0])
    index = np.flatnonzero(seen)
    frames = np.arange(len(ball))
    after = np.searchsorted(index, frames)
    inside = (after > 0) & (after < len(index))
    before = index[np.clip(after - 1, 0, len(index) - 1)]
    nxt = index[np.clip(after, 0, len(index) - 1)]
    valid = seen | (inside & (nxt - before - 1 <= MAX_GAP))

    filled = np.full_like(ball, np.nan)
    for axis in range(2):
        filled[valid, axis] = np.interp(frames[valid], index, ball[index, axis])
    return filled, valid


def smooth(ball, valid):
    """Savitzky-Golay smoothing of each continuous stretch of ball positions."""
    out = ball.copy()
    for start, stop in runs(valid):
        if stop - start >= SMOOTH_WINDOW:
            out[start:stop] = savgol_filter(ball[start:stop], SMOOTH_WINDOW, SMOOTH_ORDER, axis=0)
    return out


def ball_speed(ball):
    """Speed in m/s from a central difference; NaN next to a gap."""
    speed = np.full(len(ball), np.nan)
    speed[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * FPS / 2
    return speed


def ball_kink(ball):
    """Size of the velocity change at each frame in m/s; NaN next to a gap."""
    k = KINK_SPAN
    kink = np.full(len(ball), np.nan)
    before = ball[k:-k] - ball[:-2 * k]
    after = ball[2 * k:] - ball[k:-k]
    kink[k:-k] = np.linalg.norm(after - before, axis=1) * FPS / k
    return kink


def attack_sign(players, slot_team, half):
    """+1 if the home team attacks towards increasing x in this half, else -1.

    The defending side of a team is where its players sit on average over the half.
    """
    start, stop = half
    x = players[start:stop, :, 0]
    home = np.nanmean(x[:, slot_team == HOME])
    away = np.nanmean(x[:, slot_team == AWAY])
    return 1 if home < away else -1
