# Second-Ball

Ball possession labels for More than a Hack 2027: one of `H`, `A`, `D` or `X` for each of the
199,319 frames of the tactical video of FC Barcelona vs Athletic Club.

The method uses the tactical tracking and the Smart Tagging file only. A LightGBM classifier,
trained on the three public Metrica sample games and the 64 matches of the PFF FC World Cup 2022
data, turns per-frame tracking features into evidence, and a three-state hidden Markov model
decodes it into labels. See [TECHNICAL.md](TECHNICAL.md).

## Reproducing the submission

Requires Python 3.12 and the challenge repository cloned next to this one.

```
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git clone https://github.com/aquila147/Second-Ball
cd Second-Ball
pip install -r requirements.txt
```

Point `MTAH_DATA_DIR` at the match data bundle (the folder containing `tactical/`), then run:

```
python make_submission.py
```

This writes `Second-Ball.csv` using the trained model committed in `models/`; nothing is
retrained. The run is deterministic and takes a few minutes. If the
challenge repository is somewhere else, set `MTAH_REPO` to its path.

Check the file with the challenge validator:

```
python ../more-than-a-hack-2027/evaluation/validate_submission.py Second-Ball.csv
```

## Rule version and model training

`python make_submission.py --rules` writes the earlier version, in which hand-set rules replace
the classifier.

```
python validate.py              # leave-one-game-out score of the rules on the noised sample games
python validate.py --learned    # the same for the learned emissions
python validate.py --seed 11    # the same with another draw of the noise model
python convert_pff.py ../pff-data   # convert the PFF World Cup 2022 data (only needed to retrain)
python train.py                 # rebuild models/ from the sample games and the PFF data
python broadcast_gate.py        # the broadcast-tracking check behind dropping the broadcast
```

The PFF data is free after registering with Gradient Sports (formerly PFF FC); it is not included
here.

## Other commands

```
python diagnostics.py                                             # summary statistics of the match data
python diagnostics.py ../sample-data/data/Sample_Game_3 --noised  # the same for a noised sample game
python evaluate_sample.py ../sample-data/data/Sample_Game_3       # run and score on sample game 3
python -m unittest discover tests                                 # tests, on sample game 3
```

Everything that touches the sample games needs the public
[Metrica sample data](https://github.com/metrica-sports/sample-data) cloned next to this
repository.

## Layout

```
possession/
  io.py          tracking, metadata and Smart Tagging loaders
  clean.py       ball gap filling, smoothing, attack direction
  features.py    per-frame features
  touches.py     touch detection
  control.py     evidence for which team has the ball
  deadball.py    evidence that play is stopped
  decode.py      Viterbi decoding and boundary snapping
  pipeline.py    the steps in order, and the CSV writer
  sample.py      sample game loaders and rule-derived reference labels
  noise.py       makes sample tracking resemble the match tracking
  learned.py     LightGBM emissions: features, training, loading
  pff.py         PFF World Cup 2022 conversion and reference labels
models/          trained emissions
make_submission.py
train.py
convert_pff.py
broadcast_gate.py
validate.py
diagnostics.py
evaluate_sample.py
tests/
```

## Acknowledgements

Thresholds were checked against, and the learned emissions trained on, the Metrica Sports sample
data (https://github.com/metrica-sports/sample-data) and the PFF FC FIFA World Cup 2022 data
(Gradient Sports). The loaders for the match data come from the
challenge repository.
