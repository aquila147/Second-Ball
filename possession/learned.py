"""Learned emissions: a LightGBM classifier over per-frame features, trained on the sample games."""

import json
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from scipy.ndimage import minimum_filter1d

from . import clean, control, deadball, features, touches
from .io import HOME, REPO

MODEL_DIR = os.path.join(REPO, "models")
CAP = 500        # frames; counters stop here
TOUCH_SPAN = 12     # frames either side; the match ball track spreads a kick over about a second
TOUCH_KINK = 8.0    # m/s; the 90th percentile of the velocity change at that span
TOUCH_RADIUS = 3.0  # metres; a smoothed track cuts the corner at a touch
RESTART = 125       # frames after the start of a set-piece tag at which play resumes
TEMPER = 0.5     # weight of one frame of classifier evidence; neighbouring frames are correlated
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 200, "feature_fraction": 0.8, "seed": 2027,
          "deterministic": True, "force_row_wise": True, "verbose": -1}
ROUNDS = 300


@dataclass
class Model:
    booster: lgb.Booster
    transitions: np.ndarray
    prior: np.ndarray


def since(mask):
    """Frames since the mask was last True, capped."""
    frames = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, frames, -CAP))
    return np.minimum(frames - last, CAP)


def until(mask):
    return since(mask[::-1])[::-1]


def hold(values, valid, forward=True):
    """Value at the most recent (or next) valid frame, NaN if there is none."""
    if not forward:
        return hold(values[::-1], valid[::-1])[::-1]
    index = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))
    return np.where(index >= 0, values[index], np.nan)


def rolling_min(values, window):
    filled = np.where(np.isnan(values), np.inf, values)
    out = minimum_filter1d(filled, window, mode="nearest")
    return np.where(np.isinf(out), np.nan, out)


def smooth_ball(match):
    ball, valid = clean.fill_gaps(match.ball)
    return clean.smooth(ball, valid)


def find_touches(match, feats):
    """Touches from the velocity change over half a second, which survives a smoothed ball track."""
    kink = clean.ball_kink(smooth_ball(match), TOUCH_SPAN)
    return touches.detect(feats, kink, TOUCH_KINK, TOUCH_SPAN, TOUCH_RADIUS)


def table(match, feats, touch_frames, touch_teams):
    """Feature columns by name, in a fixed order."""
    n = len(feats.valid)
    smoothed = smooth_ball(match)
    valid = feats.valid
    home = np.where(valid, np.minimum(feats.dist[:, 0], 60.0), np.nan)
    away = np.where(valid, np.minimum(feats.dist[:, 1], 60.0), np.nan)

    ball, _ = clean.fill_gaps(match.ball)
    length, width = match.pitch
    edge = np.minimum(np.minimum(ball[:, 0], length - ball[:, 0]),
                      np.minimum(ball[:, 1], width - ball[:, 1]))

    drift = np.zeros(n)
    for half in match.halves:
        sign = clean.attack_sign(match.players, match.slot_team, half)
        drift[half[0]:half[1]] = sign * feats.drift[half[0]:half[1]]

    touched = np.zeros((2, n), dtype=bool)
    touched[touch_teams, touch_frames] = True

    cols = {
        "valid": valid.astype(float),
        "since_seen": since(valid),
        "until_seen": until(valid),
        "near_home": home,
        "near_away": away,
        "advantage": away - home,
        "speed": feats.speed,
        "kink": feats.kink,
        "kink_6": clean.ball_kink(smoothed, 6),
        "kink_12": clean.ball_kink(smoothed, 12),
        "outside": np.where(valid, feats.outside, np.nan),
        "still": np.where(valid, feats.still, np.nan),
        "edge": edge,
        "player_speed": feats.player_speed,
        "drift": drift,
        "since_touch_home": since(touched[0]),
        "since_touch_away": since(touched[1]),
        "until_touch_home": until(touched[0]),
        "until_touch_away": until(touched[1]),
        "last_near_home": hold(home, valid),
        "last_near_away": hold(away, valid),
        "next_near_home": hold(home, valid, forward=False),
        "next_near_away": hold(away, valid, forward=False),
        "last_outside": hold(feats.outside, valid),
        "last_edge": hold(edge, valid),
    }
    for seconds, window in ((1, 25), (2, 51), (6, 151)):
        for name in ("near_home", "near_away", "advantage", "speed", "kink_12", "player_speed",
                     "drift", "valid"):
            cols["{}_{}s".format(name, seconds)] = features.rolling_mean(cols[name], window)
    for name in ("near_home", "near_away"):
        cols[name + "_min_2s"] = rolling_min(cols[name], 51)
    for touched_by, name in ((touched[0], "home"), (touched[1], "away")):
        cols["touches_{}_4s".format(name)] = np.convolve(touched_by, np.ones(101), mode="same")

    # Player speed just before and just after, to place the start and end of a stoppage.
    slow = features.rolling_mean(feats.player_speed, 125)
    cols["player_speed_before"] = np.concatenate((np.full(62, np.nan), slow[:-62]))
    cols["player_speed_after"] = np.concatenate((slow[62:], np.full(62, np.nan)))
    return cols


def matrix(match, feats=None):
    """(names, float32 matrix, touch frames, touch teams) for every frame."""
    if feats is None:
        feats = features.compute(match)
    touch_frames, touch_teams = find_touches(match, feats)
    cols = table(match, feats, touch_frames, touch_teams)
    data = np.stack(list(cols.values()), axis=1).astype(np.float32)
    return list(cols), data, touch_frames, touch_teams


def mirror(names, data, labels):
    """The same frames seen from the other team: swap home and away, negate signed columns."""
    swapped = [name.replace("home", "\0").replace("away", "home").replace("\0", "away")
               for name in names]
    out = data[:, [names.index(name) for name in swapped]].copy()
    signed = [i for i, name in enumerate(names) if name.startswith(("advantage", "drift"))]
    out[:, signed] *= -1
    return out, np.where(labels < 2, 1 - labels, labels)


def count_transitions(labels, halves):
    counts = np.ones((3, 3))
    for start, stop in halves:
        part = labels[start:stop]
        keep = (part[:-1] < 3) & (part[1:] < 3)
        np.add.at(counts, (part[:-1][keep], part[1:][keep]), 1)
    return counts


def examples(match, labels, step=1):
    """(names, rows, targets, transition counts) for one match, with its mirror image.

    `step` keeps every step-th graded frame; neighbouring frames carry nearly the same features.
    """
    names, data, _, _ = matrix(match)
    graded = np.flatnonzero(labels < 3)[::step]
    flipped, flipped_labels = mirror(names, data, labels)
    rows = np.concatenate([data[graded], flipped[graded]])
    targets = np.concatenate([labels[graded], flipped_labels[graded]])
    counts = count_transitions(labels, match.halves) + count_transitions(flipped_labels, match.halves)
    return names, rows, targets, counts


def fit(names, rows, targets, counts):
    """Classifier and transition matrix from stacked examples."""
    booster = lgb.train(PARAMS, lgb.Dataset(rows, targets, feature_name=names), ROUNDS)
    transitions = counts / counts.sum(axis=1, keepdims=True)
    prior = np.bincount(targets, minlength=3) / len(targets)
    return Model(booster, transitions, prior)


def train(games, extra=()):
    """Fit on [(name, match, labels)] plus already computed examples from `extra`."""
    parts = [examples(match, labels) for _, match, labels in games] + list(extra)
    names = parts[0][0]
    return fit(names, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
               sum(p[3] for p in parts))


def emissions(model, match, feats):
    """Log scores for H, A and D, and the touches used: tempered classifier evidence plus the
    Smart Tagging terms."""
    _, data, touch_frames, touch_teams = matrix(match, feats)
    scores = TEMPER * (np.log(model.booster.predict(data) + 1e-6) - np.log(model.prior))
    phase = control.phases(match, len(scores))
    scores[:, 0] += phase / 2
    scores[:, 1] -= phase / 2
    scores[:, 2] += deadball.set_pieces(match, len(scores), RESTART)
    return scores, touch_frames, touch_teams


def save(model, folder=MODEL_DIR):
    os.makedirs(folder, exist_ok=True)
    model.booster.save_model(os.path.join(folder, "emissions.txt"))
    with open(os.path.join(folder, "emissions.json"), "w", newline="\n") as handle:
        json.dump({"transitions": model.transitions.tolist(), "prior": model.prior.tolist()},
                  handle, indent=1)


def load(folder=MODEL_DIR):
    with open(os.path.join(folder, "emissions.json")) as handle:
        meta = json.load(handle)
    booster = lgb.Booster(model_file=os.path.join(folder, "emissions.txt"))
    return Model(booster, np.array(meta["transitions"]), np.array(meta["prior"]))
