"""Degrade clean sample tracking so it resembles the match tracking.

Every setting is fitted to whole-match summary statistics of the match tracking (the output of
diagnostics.py) and to nothing else. None is chosen by validation score.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import clean
from .io import AWAY, HOME, Match

TRAVEL = 0.85            # share of a stoppage the ball spends getting to the restart spot
OUT_ZONE = 3.0           # metres from a line; a stoppage starting here is a ball out of play
EXCURSION_SHARE = 0.30   # share of those in which the ball is tracked well beyond the line
EXCURSION = 10.0         # mean metres it then travels beyond the line
WANDER = 2.0             # metres the ball is moved about while play is stopped
WANDER_SMOOTH = 20.0     # frames
BALL_SMOOTH = 5.5       # frames, Gaussian sigma; the match ball track has no sharp kicks
WOBBLE = 0.40            # metres of slow wander added to a moving ball
WOBBLE_SMOOTH = 6.0      # frames
WOBBLE_SPEED = 2.0       # m/s at which the wander reaches half strength; a resting ball is steady
BLOCK_MIN = 125          # frames; the match has almost no shorter ball dropouts
BLOCK_MEAN = 300         # mean frames beyond the minimum
BLOCK_RATE_OUT = 1 / 30     # dropouts starting per frame of stoppage with the ball off the pitch
BLOCK_RATE_DEAD = 1 / 400   # per frame of any other stoppage
BLOCK_RATE_LIVE = 1 / 3800  # and per frame of live play

SLOTS_PER_TEAM = 23
TRACK_LIFE = (600, 12000)  # mean frames of short-lived and long-lived tracks, half each
TRACK_GAP = 40            # mean frames a lost player stays untracked
COLLAPSE_RATE = 1 / 4000  # tracking collapses starting per frame
COLLAPSE_LIFE = 300       # mean frames
COLLAPSE_SHARE = 0.45     # share of players lost in a collapse
PLAYER_JITTER = 0.02      # metres


def fill_dead_ball(ball, halves, pitch, rng):
    """Give the ball a path during stoppages: out past the line if it left, then to the restart."""
    out = ball.copy()
    for start, stop in halves:
        part = out[start:stop]
        seen = ~np.isnan(part[:, 0])
        first, last = np.flatnonzero(seen)[[0, -1]]
        part[:first], part[last + 1:] = part[first], part[last]
        for a, b in clean.runs(np.isnan(part[:, 0])):
            p0, p1 = part[a - 1], part[b]
            n = b - a
            travel = max(int(n * TRAVEL), 1)
            u = np.minimum(np.arange(1, n + 1) / travel, 1.0)[:, None]
            wander = gaussian_filter1d(rng.normal(0.0, 1.0, (n, 2)), WANDER_SMOOTH, axis=0,
                                       mode="nearest")
            path = p0 + (p1 - p0) * u + WANDER * np.sin(np.pi * u) * wander / wander.std()

            gaps = np.array([p0[0], pitch[0] - p0[0], p0[1], pitch[1] - p0[1]])
            if gaps.min() < OUT_ZONE and rng.random() < EXCURSION_SHARE:
                normal = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])[gaps.argmin()]
                path = path + normal * rng.exponential(EXCURSION) * np.sin(np.pi * u)
            part[a:b] = path
    return out


def dropout(dead, off_pitch, rng):
    """Mask of long ball dropouts, starting more often during stoppages than in live play."""
    n = len(dead)
    rate = np.where(dead, np.where(off_pitch, BLOCK_RATE_OUT, BLOCK_RATE_DEAD), BLOCK_RATE_LIVE)
    hazard = np.cumsum(rate)
    mask = np.zeros(n, dtype=bool)
    frame = 0
    while frame < n:
        frame = np.searchsorted(hazard, hazard[frame] + rng.exponential(1.0))
        length = BLOCK_MIN + int(rng.exponential(BLOCK_MEAN))
        mask[frame:frame + length] = True
        frame += length + 1
    return mask


def collapse(players, rng):
    """Drop a large share of the players at once for a while, as the match tracking does."""
    n = len(players)
    frame = int(rng.exponential(1 / COLLAPSE_RATE))
    while frame < n:
        length = int(rng.exponential(COLLAPSE_LIFE)) + 1
        lost = rng.random(players.shape[1]) < COLLAPSE_SHARE
        players[frame:frame + length, lost] = np.nan
        frame += length + int(rng.exponential(1 / COLLAPSE_RATE))


def fragment(players, slot_team, rng):
    """Cut each player's track into tracklets and deal them into anonymous slots."""
    n = len(players)
    out = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan)
    for team in (HOME, AWAY):
        pieces = []
        for column in np.flatnonzero(slot_team == team):
            for first, last in clean.runs(~np.isnan(players[:, column, 0])):
                frame = first
                while frame < last:
                    life = rng.exponential(TRACK_LIFE[rng.integers(2)])
                    stop = min(frame + int(life) + 1, last)
                    pieces.append((frame, stop, column))
                    frame = stop + int(rng.exponential(TRACK_GAP))

        free_at = np.zeros(SLOTS_PER_TEAM, dtype=np.int64)
        for start, stop, column in sorted(pieces):
            free = np.flatnonzero(free_at <= start)
            if len(free):
                out[start:stop, team * SLOTS_PER_TEAM + free[0]] = players[start:stop, column]
                free_at[free[0]] = stop
    return out


def degrade(match, seed):
    """A copy of the match with a match-like ball track, ball dropout and track fragmentation."""
    rng = np.random.default_rng(seed)
    dead = np.isnan(match.ball[:, 0])

    ball = fill_dead_ball(match.ball, match.halves, match.pitch, rng)
    for start, stop in match.halves:
        wobble = gaussian_filter1d(rng.normal(0.0, 1.0, (stop - start, 2)), WOBBLE_SMOOTH, axis=0)
        smooth = gaussian_filter1d(ball[start:stop], BALL_SMOOTH, axis=0)
        speed = clean.ball_speed(smooth)
        speed[[0, -1]] = 0.0
        strength = (speed / (speed + WOBBLE_SPEED))[:, None]
        ball[start:stop] = smooth + WOBBLE * strength * wobble / wobble.std()
    off_pitch = np.any((ball < 0) | (ball > match.pitch), axis=1)
    ball[dropout(dead, off_pitch, rng)] = np.nan
    for start, stop in zip([0] + [h[1] for h in match.halves], [h[0] for h in match.halves] + [len(ball)]):
        ball[start:stop] = np.nan

    players = match.players.copy()
    collapse(players, rng)
    players = fragment(players, match.slot_team, rng)
    players += rng.normal(0.0, PLAYER_JITTER, players.shape)
    slot_team = np.repeat([HOME, AWAY], SLOTS_PER_TEAM).astype(np.int8)
    return Match(players, ball, slot_team, match.halves, match.pitch, [])
