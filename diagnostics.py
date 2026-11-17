"""Whole-match summary statistics of the tracking and Smart Tagging. No per-frame output.

    python diagnostics.py                                        # challenge match
    python diagnostics.py ../sample-data/data/Sample_Game_3      # sample game 3
"""

import sys

import numpy as np

from possession import clean, control, features, io, touches

PCT = [5, 25, 50, 75, 95, 99]


def pct(values):
    return "  ".join("{:.2f}".format(v) for v in np.nanpercentile(values, PCT))


def histogram(lengths, edges):
    counts, _ = np.histogram(lengths, bins=edges)
    return "  ".join("{}-{}: {}".format(a, b - 1, c) for a, b, c in zip(edges[:-1], edges[1:], counts))


if __name__ == "__main__":
    match = io.load_sample(sys.argv[1]) if len(sys.argv) > 1 else io.load_match()
    graded = np.zeros(len(match.ball), dtype=bool)
    for start, stop in match.halves:
        graded[start:stop] = True
    feats = features.compute(match)
    seen = ~np.isnan(match.ball[:, 0])
    print("percentiles are {}".format(PCT))
    print("graded frames {}  ball seen {:.1%}  usable after gap filling {:.1%}".format(
        graded.sum(), seen[graded].mean(), feats.valid[graded].mean()))

    gaps = [b - a for a, b in clean.runs(graded & ~seen)]
    print("ball-missing runs (frames): " + histogram(gaps, [1, 6, 11, 26, 51, 126, 251, 100000]))

    length, width = match.pitch
    out = feats.outside[graded & feats.valid]
    print("ball beyond the lines {:.1%}  by more than 0.5 m {:.1%}  overshoot pct (m): {}".format(
        (out > 0).mean(), (out > 0.5).mean(), pct(out[out > 0]) if (out > 0).any() else "-"))
    print("ball speed pct (m/s): " + pct(feats.speed[graded]))
    print("ball at rest for over 2 s: {:.1%} of usable frames".format(
        (feats.still[graded & feats.valid] > 50).mean()))
    print("nearest player to ball pct (m): " + pct(feats.dist[graded & feats.valid].min(axis=1)))

    tracked = ~np.isnan(match.players[:, :, 0])
    live = tracked[graded].sum(axis=1)
    print("tracked slots per frame pct: " + pct(live))
    lifetimes = [b - a for slot in range(tracked.shape[1])
                 for a, b in clean.runs(tracked[:, slot] & graded)]
    print("track lifetime pct (frames): {}   tracks: {}".format(pct(lifetimes), len(lifetimes)))

    close = 0
    for team in (io.HOME, io.AWAY):
        p = match.players[graded][::25][:, match.slot_team == team]
        d = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=3)
        close += np.sum(d < 0.5) - np.sum(tracked[graded][::25][:, match.slot_team == team])
    print("same-team pairs within 0.5 m per sampled frame: {:.3f}".format(
        close / 2 / len(match.players[graded][::25])))

    print("mean player speed pct (m/s): " + pct(feats.player_speed[graded]))
    for half in match.halves:
        print("half {}: home attacks towards {} x".format(
            half, "increasing" if clean.attack_sign(match.players, match.slot_team, half) > 0
            else "decreasing"))

    frames, teams = touches.detect(feats)
    inside = graded[frames]
    print("touches {}  home share {:.2f}".format(inside.sum(), (teams[inside] == io.HOME).mean()))

    if match.tags:
        by_team = [sum(tag.team == t for tag in match.tags) for t in (io.HOME, io.AWAY, -1)]
        print("tags: home {}  away {}  no team {}".format(*by_team))
        pieces = [tag for tag in match.tags if tag.code == "set pieces"]
        print("set pieces {}  duration pct (s): {}".format(
            len(pieces), pct([(tag.stop - tag.start) / io.FPS for tag in pieces])))
        for kind in sorted({tag.kind for tag in pieces}):
            print("  {}: {}".format(kind, sum(tag.kind == kind for tag in pieces)))

        n = len(match.ball)
        attack = np.zeros((2, n), dtype=bool)
        defend = np.zeros((2, n), dtype=bool)
        for tag in match.tags:
            if tag.team >= 0 and tag.code in control.ATTACKING_CODES:
                attack[tag.team, tag.start:tag.stop] = True
            if tag.team >= 0 and tag.code in control.DEFENDING_CODES:
                defend[tag.team, tag.start:tag.stop] = True
        for start, stop in match.halves:
            half = slice(start, stop)
            print("half ({}, {}): attacking tag on {:.1%} of frames, defending tag on {:.1%}".format(
                start, stop, attack[:, half].any(axis=0).mean(), defend[:, half].any(axis=0).mean()))
        # If Team names the acting side, one side's defending tags sit under the other's attacking tags.
        same = (attack[0] & defend[0]) | (attack[1] & defend[1])
        cross = (attack[0] & defend[1]) | (attack[1] & defend[0])
        print("frames with attacking and defending tags: opposite teams {}  same team {}".format(
            cross[graded].sum(), same[graded].sum()))
        both = attack[0] & attack[1]
        print("frames with attacking tags for both teams: {}".format(both[graded].sum()))
