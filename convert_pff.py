"""Convert the PFF FC World Cup 2022 tracking to 25 fps arrays with reference labels.

    python convert_pff.py C:/hackathon/pff-data

Writes <folder>/cache/<game_id>.npz. Matches already converted are skipped unless --force.
"""

import argparse
import os
from multiprocessing import Pool

from possession import pff


def work(args):
    folder, game_id = args
    return pff.convert(folder, game_id, os.path.join(folder, "cache"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="convert matches already converted")
    args = parser.parse_args()

    cache = os.path.join(args.folder, "cache")
    todo = [(args.folder, g) for g in pff.game_ids(args.folder)
            if args.force or not os.path.exists(os.path.join(cache, "{}.npz".format(g)))]
    print("{} matches to convert".format(len(todo)), flush=True)
    with Pool(args.processes) as pool:
        for done, game_id in enumerate(pool.imap_unordered(work, todo), 1):
            print("{}/{} {}".format(done, len(todo), game_id), flush=True)
