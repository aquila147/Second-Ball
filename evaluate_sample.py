"""Run the pipeline on public sample game 3 and score it against rule-derived labels.

    python evaluate_sample.py ../sample-data/data/Sample_Game_3

The reference labels come from the sample's own tracking and events (possession/sample.py).
Set pieces from the event file stand in for the Smart Tagging restarts with --set-pieces.
"""

import argparse
import json
import os
import sys

from possession import io, pipeline, sample

KINDS = {"THROW IN": "throw-in", "GOAL KICK": "goal kick", "CORNER KICK": "corner kick",
         "FREE KICK": "free kick", "KICK OFF": "kick off (after goal)"}


def set_piece_tags(folder, home_id="FIFATMA"):
    """Tag-like intervals of 5 s around each SET PIECE event."""
    stem = os.path.join(folder, os.path.basename(os.path.normpath(folder)))
    with open(stem + "_events.json") as handle:
        events = json.load(handle)["data"]
    tags = []
    for event in events:
        if event["type"]["name"] != "SET PIECE" or not event["subtypes"]:
            continue
        subtypes = event["subtypes"] if isinstance(event["subtypes"], list) else [event["subtypes"]]
        kind = next((KINDS[s["name"]] for s in subtypes if s["name"] in KINDS), None)
        if kind:
            frame = event["start"]["frame"] - 1
            team = io.HOME if event["team"]["id"] == home_id else io.AWAY
            tags.append(io.Tag("set pieces", kind, team, frame - 62, frame + 63))
    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--set-pieces", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(io.CHALLENGE_REPO, "evaluation"))
    import possession_metrics as pm

    match = io.load_sample(args.folder)
    if args.set_pieces:
        match.tags = set_piece_tags(args.folder)
    truth = sample.reference_labels(args.folder, match)
    codes = pipeline.label_codes(match)

    print("\n".join(pipeline.summary(codes)))
    result = pm.evaluate(truth, codes)
    print("spellF1 {:.4f}  macroF1 {:.4f}".format(result["spell_f1_mean"], result["macro_f1"]))
    print("possessions: reference {}  predicted {}".format(
        result["n_gt_spells"], result["n_pred_spells"]))
    print("team accuracy where both alive {:.3f}  alive/dead F1 {:.3f}".format(
        result["team_acc_both_alive"], result["alive_dead_f1"]))
    for name, row in result["per_class"].items():
        print("{}  precision {:.3f}  recall {:.3f}".format(name, row["precision"], row["recall"]))
