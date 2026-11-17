"""Write the possession submission for the challenge match.

    python make_submission.py                  # writes Second-Ball.csv
    python make_submission.py --out other.csv

Reads the match data from MTAH_DATA_DIR and the challenge repo from MTAH_REPO
(default: ../more-than-a-hack-2027).
"""

import argparse

from possession import io, pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="Second-Ball.csv")
    args = parser.parse_args()

    codes = pipeline.label_codes(io.load_match())
    pipeline.write_submission(args.out, codes)
    print("wrote {}".format(args.out))
    print("\n".join(pipeline.summary(codes)))
