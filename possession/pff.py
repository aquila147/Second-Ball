"""PFF FC World Cup 2022 data: conversion to 25 fps arrays and rule-derived reference labels.

Labels come from the game events embedded in the tracking, with dead ball from an OUT event until
the next on-the-ball event and X outside the halves. Two possession rules are stored: "sequence",
the team of PFF's current possession sequence (`home_ball`), and "touch", the team of the player on
the ball (`home_team`), which switches at every on-the-ball event like the Metrica sample labels.
"""

import bz2
import json
import os

import numpy as np

from .io import AWAY, FPS, HOME, Match

H, A, D, X = 0, 1, 2, 3
SLOTS_PER_TEAM = 23


def tracking_path(folder, game_id):
    """Compressed file, or the folder an archive tool unpacked it into."""
    packed = os.path.join(folder, "Tracking Data", "{}.jsonl.bz2".format(game_id))
    if os.path.exists(packed):
        return packed
    return os.path.join(folder, "Tracking Data", "{}.jsonl".format(game_id),
                        "{}.jsonl".format(game_id))


def game_ids(folder):
    return sorted(int(name.split(".")[0]) for name in os.listdir(os.path.join(folder, "Metadata"))
                  if name.endswith(".json"))


def read_frames(path):
    """(times, players, player visible, ball, ball visible, states, period) per frame.

    `states` holds the event state per frame for the sequence and touch rules, -1 between events.

    The period is 0 for frames before a half's clock has started.
    """
    opener = bz2.open if path.endswith(".bz2") else open
    times, rows, seen, balls, ball_seen, state, touch, periods = [], [], [], [], [], [], [], []
    slot_of = [{}, {}]
    with opener(path, "rt") as handle:
        for line in handle:
            frame = json.loads(line)
            times.append(frame["videoTimeMs"] / 1000.0)
            periods.append(frame.get("period") or 0 if (frame.get("periodElapsedTime") or 0) > 0 else 0)
            row = np.full((2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
            visible = np.zeros(2 * SLOTS_PER_TEAM, dtype=bool)
            for team, key in ((HOME, "homePlayers"), (AWAY, "awayPlayers")):
                for player in frame.get(key) or []:
                    if player.get("x") is None:
                        continue
                    slots = slot_of[team]
                    slot = slots.setdefault(player["jerseyNum"], len(slots))
                    if slot >= SLOTS_PER_TEAM:
                        continue
                    row[team * SLOTS_PER_TEAM + slot] = (player["x"], player["y"])
                    visible[team * SLOTS_PER_TEAM + slot] = player["visibility"] == "VISIBLE"
            rows.append(row)
            seen.append(visible)

            ball = (frame.get("balls") or [None])[0]
            if ball and ball.get("x") is not None:
                balls.append((ball["x"], ball["y"]))
                ball_seen.append(ball["visibility"] == "VISIBLE")
            else:
                balls.append((np.nan, np.nan))
                ball_seen.append(False)

            event = frame.get("game_event") or {}
            kind = event.get("game_event_type")
            if kind == "OUT":
                state.append(D)
                touch.append(D)
            elif kind == "OTB":
                ball_team, player_team = event.get("home_ball"), event.get("home_team")
                state.append(-1 if ball_team is None else H if ball_team else A)
                touch.append(-1 if player_team is None else H if player_team else A)
            else:
                state.append(-1)
                touch.append(-1)
    return (np.array(times), np.array(rows), np.array(seen), np.array(balls, dtype=np.float32),
            np.array(ball_seen), np.array([state, touch], dtype=np.int8).T, np.array(periods))


def convert(folder, game_id, out_folder):
    """Write one match as 25 fps arrays to out_folder/<game_id>.npz."""
    with open(os.path.join(folder, "Metadata", "{}.json".format(game_id))) as handle:
        meta = json.load(handle)[0]
    pitch = meta["stadium"]["pitches"][0]
    length, width = float(pitch["length"]), float(pitch["width"])

    times, players, seen, ball, ball_seen, state, periods = read_frames(
        tracking_path(folder, game_id))
    order = np.argsort(times, kind="stable")
    times, players, seen, ball, ball_seen, state, periods = (
        a[order] for a in (times, players, seen, ball, ball_seen, state, periods))

    # Carry the last event forward: a team keeps the ball, and play stays dead, until the next event.
    filled = np.empty_like(state)
    for rule in range(2):
        column = state[:, rule]
        carry = np.maximum.accumulate(np.where(column >= 0, np.arange(len(column)), -1))
        filled[:, rule] = np.where(carry >= 0, column[np.maximum(carry, 0)], D)

    halves_s = [(meta["startPeriod1"], meta["endPeriod1"]), (meta["startPeriod2"], meta["endPeriod2"])]
    if None in sum(halves_s, ()):
        # Some metadata files leave the half times out; take them from the tracking clock.
        halves_s = [(times[periods == p].min(), times[periods == p].max()) for p in (1, 2)]
    grid = np.arange(halves_s[0][0], halves_s[1][1], 1.0 / FPS)
    nearest = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    nearest -= (grid - times[nearest - 1]) < (times[nearest] - grid)
    gap = np.abs(times[nearest] - grid) > 0.5 / FPS * 2

    labels = filled[nearest].astype(np.int8)
    labels[gap] = X
    halves = []
    for start_s, stop_s in halves_s:
        first, stop = np.searchsorted(grid, [start_s, stop_s])
        halves.append((int(first), int(stop)))
    inside = np.zeros(len(grid), dtype=bool)
    for first, stop in halves:
        inside[first:stop] = True
    labels[~inside] = X

    offset = np.array([length / 2, width / 2], dtype=np.float32)
    rows = players[nearest] + offset
    rows[gap] = np.nan
    ball_rows = ball[nearest] + offset
    ball_rows[gap] = np.nan

    os.makedirs(out_folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_folder, "{}.npz".format(game_id)),
        players=rows, players_visible=seen[nearest] & ~gap[:, None], ball=ball_rows,
        ball_visible=ball_seen[nearest] & ~gap, labels=labels[:, 0], labels_touch=labels[:, 1],
        halves=np.array(halves), pitch=np.array([length, width]))
    return game_id


def load(cache_folder, game_id, rule="sequence"):
    """(match, labels, ball_visible, players_visible) from a converted match."""
    data = np.load(os.path.join(cache_folder, "{}.npz".format(game_id)))
    slot_team = np.repeat([HOME, AWAY], SLOTS_PER_TEAM).astype(np.int8)
    halves = [tuple(int(v) for v in pair) for pair in data["halves"]]
    match = Match(data["players"].astype(float), data["ball"].astype(float), slot_team, halves,
                  tuple(float(v) for v in data["pitch"]), [])
    labels = data["labels"] if rule == "sequence" else data["labels_touch"]
    return match, labels, data["ball_visible"], data["players_visible"]


def training_rows(cache_folder, game_id, seed, step=10, rule="touch"):
    """Noised training examples for one converted match: (names, rows, targets, counts).

    Only every `step`-th frame is kept: neighbouring frames carry nearly the same features.
    """
    from . import learned, noise

    match, labels, _, _ = load(cache_folder, game_id, rule)
    noised = noise.degrade(match, seed * 7919 + game_id, dead=labels == D)
    names, rows, targets, counts = learned.examples(noised, labels, step)
    return names, rows.astype(np.float32), targets, counts
