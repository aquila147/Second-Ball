"""Tracking in, one label per frame out."""

import numpy as np

from . import control, deadball, decode, features, touches

LABELS = np.array(["H", "A", "D", "X"])
X = 3


def label_codes(match):
    """Codes into LABELS for every frame: X outside the halves, a decoded state inside."""
    feats = features.compute(match)
    touch_frames, touch_teams = touches.detect(feats)
    scores = decode.emissions(control.team_evidence(match, feats, touch_frames, touch_teams),
                              deadball.dead_evidence(match, feats))

    codes = np.full(len(scores), X, dtype=np.int8)
    for start, stop in match.halves:
        inside = (touch_frames >= start) & (touch_frames < stop)
        path = decode.viterbi(scores[start:stop])
        codes[start:stop] = decode.snap(path, touch_frames[inside] - start, touch_teams[inside])
    return codes


def write_submission(path, codes):
    """frame,label CSV with LF line endings on every platform."""
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(LABELS[codes]))


def summary(codes, fps=25):
    """Label counts and possession statistics as printable lines."""
    lines = ["{}  {:>7}".format(label, int(np.sum(codes == i))) for i, label in enumerate(LABELS)]
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(codes)]))
    lengths = np.array([b - a for a, b in zip(bounds[:-1], bounds[1:]) if codes[a] < 2])
    lines.append("possessions {}  mean {:.1f} s  median {:.1f} s".format(
        len(lengths), lengths.mean() / fps, np.median(lengths) / fps))
    return lines
