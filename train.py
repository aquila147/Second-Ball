"""Train the learned emissions and save them to models/.

    python train.py

Training data: the three public Metrica sample games and the 64 PFF FC World Cup 2022 matches,
all degraded by the noise model. The PFF matches must first be converted with convert_pff.py;
their noised training rows are cached next to the converted files. The trained model is
committed, so this is only needed to rebuild it.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

import validate
from possession import io, learned, pff

PFF_SEED = validate.SEED


def rows(args):
    """Training rows for one PFF match, from the cache if they were built before."""
    cache, game_id = args
    path = os.path.join(cache, "rows_touch_{}".format(PFF_SEED), "{}.npz".format(game_id))
    if not os.path.exists(path):
        names, data, targets, counts = pff.training_rows(cache, game_id, PFF_SEED)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, rows=data, targets=targets.astype(np.int8), counts=counts,
                 names=np.array(names))
    stored = np.load(path)
    return (list(stored["names"]), stored["rows"], stored["targets"].astype(np.int64),
            stored["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(io.REPO),
                                                          "sample-data", "data"))
    parser.add_argument("--pff", default=os.path.join(os.path.dirname(io.REPO), "pff-data"))
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args()

    cache = os.path.join(args.pff, "cache")
    ids = sorted(pff.game_ids(args.pff), key=str)
    with Pool(args.processes) as pool:
        extra = pool.map(rows, [(cache, game_id) for game_id in ids])
    learned.save(learned.train(validate.load_games(args.samples), extra))
    print("wrote {}".format(learned.MODEL_DIR))
