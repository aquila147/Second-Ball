"""Leave-one-game-out validation on the noised public sample games, scored with the official metric.

    python validate.py                 # Phase 1 rules; nothing is trained
    python validate.py --learned       # LightGBM emissions, trained on the other two games
    python validate.py --clean         # Phase 1 on the sample tracking as published
    python validate.py --seed 7        # the same with another draw of the noise model

Expects ../sample-data (or --samples) and the challenge repo (MTAH_REPO).
"""

import argparse
import os
import sys

import numpy as np

from possession import io, noise, pipeline, sample

GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
SEED = 2027


def load_games(root, clean=False, seed=SEED):
    """[(name, match, labels)] with the noise model applied unless clean."""
    games = []
    for index, name in enumerate(GAMES):
        match, labels = sample.load(os.path.join(root, name))
        if not clean:
            match = noise.degrade(match, seed + index)
        games.append((name, match, labels))
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(io.REPO),
                                                          "sample-data", "data"))
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="noise model seed")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(io.CHALLENGE_REPO, "evaluation"))
    import possession_metrics as pm

    games = load_games(args.samples, args.clean, args.seed)
    spell, macro = [], []
    for held_out, (name, match, labels) in enumerate(games):
        model = None
        if args.learned:
            from possession import learned
            model = learned.train([g for i, g in enumerate(games) if i != held_out])
        codes = pipeline.label_codes(match, model)
        result = pm.evaluate(labels, codes)
        spell.append(result["spell_f1_mean"])
        macro.append(result["macro_f1"])
        print("{}  spellF1 {:.4f}  macroF1 {:.4f}  possessions {} (reference {})  "
              "D recall {:.2f}  team accuracy {:.3f}".format(
                  name, spell[-1], macro[-1], result["n_pred_spells"], result["n_gt_spells"],
                  result["per_class"]["D"]["recall"], result["team_acc_both_alive"]))
    print("mean           spellF1 {:.4f}  macroF1 {:.4f}".format(np.mean(spell), np.mean(macro)))
