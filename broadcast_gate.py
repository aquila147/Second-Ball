"""Can the broadcast tracking fill the tactical ball gaps? Whole-match summary statistics only.

    python broadcast_gate.py

Each continuous broadcast shot is aligned to the tactical timeline by matching ball positions
(masked least squares over every offset, computed with FFTs) and checked by matching player
positions at the chosen offset. Prints how many shots align, how well the two ball tracks agree,
and how much of the tactical ball gaps the aligned broadcast covers.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import fftconvolve

from possession import clean, io

MIN_SHOT = 250        # frames; shorter shots are not aligned
CUT_JUMP = 150.0      # pixels the pitch centre may move between frames within one shot
MIN_OVERLAP = 75      # frames with the ball in both sources at the chosen offset
PLAYER_STEP = 5       # check players on every fifth frame of a shot
ACCEPT_PLAYERS = 1.5  # metres, median distance from a broadcast player to the nearest tactical one
ACCEPT_BALL = 2.0     # metres, root mean square ball disagreement

PCT = [5, 25, 50, 75, 95]


def correlate(shot, full):
    """sum_i shot[i] * full[i + o] for every offset o."""
    return fftconvolve(full, shot[::-1], mode="valid")


def best_offset(ball_b, ball_t):
    """(offset, rms, overlap) minimising the mean squared ball distance over all offsets."""
    mb = ~np.isnan(ball_b[:, 0])
    mt = ~np.isnan(ball_t[:, 0])
    b = np.nan_to_num(ball_b)
    t = np.nan_to_num(ball_t)
    overlap = np.rint(correlate(mb.astype(float), mt.astype(float)))
    ssd = np.zeros_like(overlap)
    for axis in range(2):
        ssd += correlate(b[:, axis] ** 2, mt.astype(float))
        ssd += correlate(mb.astype(float), t[:, axis] ** 2)
        ssd -= 2 * correlate(b[:, axis], t[:, axis])
    enough = overlap >= max(MIN_OVERLAP, 0.3 * mb.sum())
    if not enough.any():
        return None
    mean = np.where(enough, ssd / np.maximum(overlap, 1), np.inf)
    offset = int(np.argmin(mean))
    return offset, float(np.sqrt(max(mean[offset], 0.0))), int(overlap[offset])


def player_distance(players_b, team_b, players_t, team_t):
    """Median distance from each broadcast player to the nearest tactical player of its team."""
    out = []
    for team in (io.HOME, io.AWAY):
        pb = players_b[:, team_b == team]
        pt = players_t[:, team_t == team]
        d = np.linalg.norm(pb[:, :, None] - pt[:, None, :], axis=3)
        nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)
        out.append(nearest[~np.isnan(pb[:, :, 0]) & np.isfinite(nearest)])
    both = np.concatenate(out)
    return float(np.median(both)) if len(both) else np.inf


def shots(valid, matrices):
    """Continuous broadcast shots: valid homography with no jump of the projected pitch centre."""
    centre = matrices @ np.array([0.5, 0.5, 1.0])
    pixel = centre[:, :2] / centre[:, 2:]
    jump = np.zeros(len(valid), dtype=bool)
    jump[1:] = np.linalg.norm(np.diff(pixel, axis=0), axis=1) > CUT_JUMP
    out = []
    for start, stop in clean.runs(valid):
        cuts = [start] + [i for i in range(start + 1, stop) if jump[i]] + [stop]
        out += [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= MIN_SHOT]
    return out


def load_source(games, source):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], source=source)
    pitch = games.GAMES[games.GAME]["pitch"]
    players, ball = io.read_tracking(raw_path, games.source_frames(source))
    slot_team = io.slot_teams(ET.parse(meta_path).getroot(), players.shape[1])
    return players * pitch, ball * pitch, slot_team


def pct(values):
    return "  ".join("{:.2f}".format(v) for v in np.percentile(values, PCT)) if len(values) else "-"


if __name__ == "__main__":
    sys.path.insert(0, io.CHALLENGE_REPO)
    import games
    from starter import load_homography

    players_t, ball_t, team_t = load_source(games, "tactical")
    players_b, ball_b, team_b = load_source(games, "broadcast")
    matrices, _, valid = load_homography.load(source="broadcast")
    pitch = np.array(games.GAMES[games.GAME]["pitch"])

    graded = np.zeros(len(ball_t), dtype=bool)
    for _, first, last in games.half_video_blocks(games.GAME):
        graded[first:last + 1] = True

    found = shots(valid, matrices)
    print("broadcast frames {}  valid homography {:.1%}  ball seen {:.1%}".format(
        len(ball_b), valid.mean(), (~np.isnan(ball_b[:, 0])).mean()))
    print("shots of at least {} frames: {}  covering {:.1%} of broadcast frames".format(
        MIN_SHOT, len(found), sum(b - a for a, b in found) / len(ball_b)))

    results = {}
    for name, sign in (("same orientation", 1), ("mirrored", -1)):
        flip = (lambda p: p) if sign > 0 else (lambda p: pitch - p)
        rows = []
        for start, stop in found:
            hit = best_offset(flip(ball_b[start:stop]), ball_t)
            if hit is None:
                continue
            offset, rms, overlap = hit
            frames = np.arange(start, stop, PLAYER_STEP)
            dist = player_distance(flip(players_b[frames]), team_b,
                                   players_t[frames - start + offset], team_t)
            rows.append((start, stop, offset, rms, overlap, dist))
        accepted = [r for r in rows if r[3] <= ACCEPT_BALL and r[5] <= ACCEPT_PLAYERS]
        results[name] = (rows, accepted)
        print("{}: shots with a candidate offset {}  accepted {}".format(name, len(rows),
                                                                         len(accepted)))

    name = max(results, key=lambda k: len(results[k][1]))
    rows, accepted = results[name]
    flip = (lambda p: p) if name == "same orientation" else (lambda p: pitch - p)
    print("\nusing {}".format(name))
    print("candidate shots, ball rms (m): " + pct([r[3] for r in rows]))
    print("candidate shots, player median distance (m): " + pct([r[5] for r in rows]))
    for ball_limit, player_limit in ((1.0, 1.0), (2.0, 1.5), (3.0, 2.5)):
        n = sum(r[3] <= ball_limit and r[5] <= player_limit for r in rows)
        print("  accepted at ball <= {} m and players <= {} m: {}".format(ball_limit, player_limit, n))

    source_of = np.full(len(ball_t), -1)
    for start, stop, offset, _, _, _ in accepted:
        target = np.arange(offset, offset + stop - start)
        free = source_of[target] < 0
        source_of[target[free]] = np.arange(start, stop)[free]
    mapped = source_of >= 0
    borrowed = np.full_like(ball_t, np.nan)
    borrowed[mapped] = flip(ball_b[source_of[mapped]])
    print("accepted shots cover {:.1%} of graded tactical frames".format(mapped[graded].mean()))

    seen_t = ~np.isnan(ball_t[:, 0])
    seen_b = ~np.isnan(borrowed[:, 0])
    both = graded & seen_t & seen_b
    print("graded frames with the ball in both: {}  distance pct (m): {}".format(
        both.sum(), pct(np.linalg.norm(ball_t[both] - borrowed[both], axis=1))))

    gaps = graded & ~seen_t
    print("\ngraded frames without a tactical ball: {}".format(gaps.sum()))
    print("  of those, an aligned broadcast frame exists: {:.1%}".format(mapped[gaps].mean()))
    print("  of those, the broadcast has the ball: {:.1%}".format(seen_b[gaps].mean()))
    inside = seen_b & np.all((borrowed >= 0) & (borrowed <= pitch), axis=1)
    print("  of those, the broadcast ball is on the pitch: {:.1%}".format(inside[gaps].mean()))
    for low, high in ((1, 25), (25, 125), (125, 250), (250, 10 ** 6)):
        in_runs = np.zeros(len(ball_t), dtype=bool)
        for a, b in clean.runs(gaps):
            if low <= b - a < high:
                in_runs[a:b] = True
        if in_runs.any():
            print("  gaps of {}-{} frames: {} frames, broadcast ball available on {:.1%}".format(
                low, high - 1, in_runs.sum(), seen_b[in_runs].mean()))
