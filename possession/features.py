"""Per-frame features from cleaned tracking."""

from dataclasses import dataclass

import numpy as np

from . import clean
from .io import AWAY, FPS, HOME

SPEED_SPAN = 5         # frames used for a player displacement
MAX_PLAYER_SPEED = 12  # m/s; anything faster is a track swap
STILL_SPEED = 0.75     # m/s; a ball slower than this is at rest, allowing for jitter


@dataclass
class Features:
    valid: np.ndarray         # ball position usable (seen or interpolated)
    dist: np.ndarray          # (F, 2) metres from ball to nearest home / away player
    speed: np.ndarray         # ball speed, m/s
    kink: np.ndarray          # ball velocity change, m/s
    outside: np.ndarray       # metres the ball is beyond the pitch boundary, 0 inside
    still: np.ndarray         # consecutive frames the ball has been at rest
    player_speed: np.ndarray  # mean player speed over 1 s, m/s
    drift: np.ndarray         # mean player velocity along x over 2 s, m/s


def rolling_mean(values, window):
    """Centred mean that ignores NaN."""
    ok = ~np.isnan(values)
    kernel = np.ones(window)
    total = np.convolve(np.where(ok, values, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def nearest(players, ball, slot_team):
    """(F, 2) distance from the ball to the nearest player of each team, inf if none."""
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    return np.stack([dist[:, slot_team == HOME].min(axis=1),
                     dist[:, slot_team == AWAY].min(axis=1)], axis=1)


def player_motion(players):
    """Mean player speed and mean x velocity per frame, NaN-aware."""
    k = SPEED_SPAN
    step = np.full_like(players, np.nan)
    step[k:] = (players[k:] - players[:-k]) * FPS / k
    speed = np.linalg.norm(step, axis=2)
    swap = speed > MAX_PLAYER_SPEED
    speed[swap] = np.nan
    vx = np.where(swap, np.nan, step[:, :, 0])

    tracked = ~np.isnan(speed)
    count = np.maximum(tracked.sum(axis=1), 1)
    mean_speed = np.where(tracked.any(axis=1), np.nansum(speed, axis=1) / count, np.nan)
    mean_vx = np.where(tracked.any(axis=1), np.nansum(vx, axis=1) / count, np.nan)
    return rolling_mean(mean_speed, FPS), rolling_mean(mean_vx, 2 * FPS)


def still_frames(speed):
    """Number of consecutive frames up to each one with the ball at rest."""
    rest = speed < STILL_SPEED
    count = np.zeros(len(speed), dtype=np.int32)
    for start, stop in clean.runs(rest):
        count[start:stop] = np.arange(1, stop - start + 1)
    return count


def compute(match):
    ball, valid = clean.fill_gaps(match.ball)
    ball = clean.smooth(ball, valid)
    speed = clean.ball_speed(ball)

    length, width = match.pitch
    beyond = np.maximum(np.maximum(-ball, ball - (length, width)), 0.0)
    outside = np.where(valid, beyond.max(axis=1), 0.0)

    player_speed, drift = player_motion(match.players)
    return Features(valid, nearest(match.players, ball, match.slot_team), speed,
                    clean.ball_kink(ball), outside, still_frames(speed), player_speed, drift)
