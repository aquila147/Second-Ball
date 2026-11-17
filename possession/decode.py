"""Three-state Viterbi decoding and boundary snapping."""

import numpy as np

from .io import FPS

H, A, D = 0, 1, 2

# Rows are the current state, columns the next. Possessions and stoppages last about 10 s, so
# a state is left about once in 250 frames; a possession ends in a turnover more often than in
# a stoppage.
TRANSITIONS = np.array([[0.996, 0.003, 0.001],
                        [0.003, 0.996, 0.001],
                        [0.002, 0.002, 0.996]])

SNAP_WINDOW = FPS  # frames either side of a decoded change to look for the new team's touch


def emissions(team, dead):
    """Log scores per frame for H, A and D from the two evidence streams."""
    return np.stack([team / 2, -team / 2, dead], axis=1)


def viterbi(scores):
    """Most likely state sequence for one half."""
    log_trans = np.log(TRANSITIONS)
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    best = scores[0].copy()
    for t in range(1, n):
        step = best[:, None] + log_trans
        back[t] = step.argmax(axis=0)
        best = step.max(axis=0) + scores[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap(path, touch_frames, touch_teams):
    """Move each change between teams to the new team's first touch after the old team's last."""
    path = path.copy()
    changes = np.flatnonzero((path[1:] != path[:-1]) & (path[1:] != D) & (path[:-1] != D)) + 1
    edges = np.concatenate(([0], np.flatnonzero(path[1:] != path[:-1]) + 1, [len(path)]))
    for change in changes:
        new, old = path[change], path[change - 1]
        i = np.searchsorted(edges, change)
        low = max(change - SNAP_WINDOW, edges[i - 1] + 1)
        high = min(change + SNAP_WINDOW, edges[i + 1] - 1)
        near = (touch_frames >= low) & (touch_frames <= high)
        frames, teams = touch_frames[near], touch_teams[near]

        old_before = frames[(teams == old) & (frames <= change)]
        after = frames[(teams == new) & (frames > (old_before.max() if len(old_before) else low - 1))]
        if len(after):
            target = after[0]
            path[min(change, target):max(change, target)] = old if target > change else new
    return path
