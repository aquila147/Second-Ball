"""Train the learned emissions on the three noised sample games and save them to models/.

    python train.py

The trained model is committed, so this is only needed to rebuild it.
"""

import argparse
import os

import validate
from possession import io, learned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(io.REPO),
                                                          "sample-data", "data"))
    args = parser.parse_args()

    learned.save(learned.train(validate.load_games(args.samples)))
    print("wrote {}".format(learned.MODEL_DIR))
