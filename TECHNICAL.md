# Second-Ball: technical document

## Approach

Every frame inside the two halves gets one of three states: home possession (`H`), away
possession (`A`) or dead ball (`D`). Frames outside the halves get `X`.

The pipeline computes per-frame evidence from the tracking and the Smart Tagging file, and a
three-state hidden Markov model turns it into a label sequence. There are two versions, which
share everything except where the evidence comes from:

- **Learned emissions (the submitted file).** A LightGBM classifier trained on the three public
  Metrica sample games gives the evidence. Described under "Learned emissions" below.
- **Rules** (`--rules`, the first submission). The evidence is built with hand-set rules,
  described in steps 1 to 9 and the next two sections.

The steps of the rule version:

1. **Load.** Player positions `(frames, 46 slots, 2)` and ball positions in metres, and the team
   of each slot from the metadata. The speed column of the tracking is not used; speeds are
   recomputed from positions so the same code runs on the public sample data.
2. **Ball cleaning.** Gaps of up to 10 frames are filled by linear interpolation. Each continuous
   stretch is smoothed with a Savitzky-Golay filter (window 7, order 2). Longer gaps stay missing.
3. **Features.** Distance from the ball to the nearest player of each team; ball speed; size of
   the ball's velocity change over 3 frames either side; distance of the ball beyond the pitch
   boundary; how long the ball has been at rest; mean player speed over 1 s; mean player velocity
   along the pitch over 2 s; attack direction per half (the team whose players sit at lower x on
   average over the half attacks towards higher x).
4. **Touches.** Peaks in the ball's velocity change (`scipy.signal.find_peaks`) with a player
   within 2 m. The touch is given to the nearest player's team.
5. **Team evidence** (log-likelihood ratio, home against away): ball at a player's feet, touches,
   collective movement of both teams, and the team named by the Smart Tagging phase covering the
   frame.
6. **Dead-ball evidence** (log-likelihood ratio, dead against alive): ball beyond the lines, ball
   at rest, slow players, the lead-in to a tagged set piece, and everything before the tagged
   kick-off of a half. A ball moving on the pitch counts against dead ball.
7. **Decoding.** Viterbi over `H`, `A`, `D`, each half separately. Emission log scores are
   `+team/2`, `-team/2` and `dead`.
8. **Boundary snapping.** Each decoded change between teams is moved to the new team's first
   touch after the old team's last touch, within 1 s of the decoded frame.
9. **Write.** `frame,label` with LF line endings.

## Possession rules (rule version)

There is one decision mechanism, the Viterbi decoder. The rules below are consequences of the
emission weights and the transition matrix.

- **Missing ball.** When the ball is not tracked, all ball-based evidence is zero, so the current
  state is held unless the Smart Tagging phase, collective movement or player speed argue
  otherwise. A missing ball is not treated as a dead ball.
- **Control.** Leaving a team's possession costs `log(0.003)`, about 5.8. A touch is worth 2.0 and
  each frame with the ball at one team's feet up to 0.6. One touch or a deflection does not change
  possession. Two touches in quick succession with the ball staying close do, as does a touch
  followed by about 0.3 s with the ball at that team's feet.
- **Loose and airborne balls** stay with the last team: with the ball further than about 2.5 m
  from everyone, or moving faster than about 12 m/s, there is no proximity evidence.
- **Dead ball.** Entering and leaving `D` costs about 13 in total, so a few slow or still frames do
  not create a stoppage. A ball more than 0.5 m beyond the lines does so within a few frames.
- **Restart.** `D` ends when the ball moves on the pitch again. The restarting team is decided by
  the same team evidence as everywhere else (nearest player, first touch, set-piece team tag).
- **Second-half lead-in.** Frames of a half before the start of its Smart Tagging
  `Kick Off (Start)` interval are forced to `D`. The kick-off frame is not hard-coded.

## Parameters (rule version)

No parameter was chosen or adjusted using the challenge match or its public ground truth. Each
comes from football reasoning or from statistics of public sample game 3.

| Parameter | Value | Reason |
|---|---|---|
| Ball gap filled | 10 frames | 0.4 s; the ball moves on a near-straight line over that time |
| Smoothing | Savitzky-Golay 7 / 2 | Removes frame-to-frame jitter and keeps a kick visible |
| Touch: velocity change | 4 m/s | Sample game 3: median 0.03, 90th percentile 3.9 m/s in live play; gives 2,079 touches in the game, 97% credited to the team in possession |
| Touch: player distance | 2 m | Reach of a player plus tracking error |
| Touch spacing | 5 frames | Two touches cannot be closer than 0.2 s |
| Ball at feet | within 2.5 m | Sample game 3: the nearer team is the one in possession in 96% of frames under 1 m, 72% at 1.5 to 2 m, 60% at 3 to 5 m |
| Controlled ball speed | under 12 m/s | A faster ball passing a player is a pass or a shot going by |
| Weight, ball at feet | 0.6 per frame | Possession changes after about 0.4 s of clear control |
| Weight, touch | 2.0 | Below the cost of a change, so one touch never decides |
| Weight, collective movement | 0.05 per frame | Sample game 3: when both teams move one way faster than 1 m/s, the team attacking that way has the ball in about 69% of frames; weak, matters only over several seconds |
| Weight, Smart Tagging phase | 0.15 per frame | Automatic tags; should be overridden by one second of ball evidence |
| Ball out | 0.5 m beyond the line | Margin for tracking error at the line |
| Ball moving | over 2 m/s | Faster than a ball being placed or rolled a short way |
| Ball at rest | under 0.75 m/s, full weight after 2 s | Sample game 3: the ball is at rest for more than 2 s in 0.4% of live frames |
| Walking speed | 1.3 m/s | Sample game 3: mean player speed has median 2.1 m/s in live play and 1.0 m/s in stoppages |
| Set-piece lead-in | throw-in 5 s; goal kick, corner, free kick 12 s; kick-off after a goal 30 s | About half the usual delay before each kind of restart |
| Stay probability | 0.996 | Possessions and stoppages last about 10 s, i.e. 250 frames |
| Turnover / stoppage | 0.003 / 0.001 | A possession ends in a turnover more often than in a stoppage |
| Restart to either team | 0.002 each | Stoppages last about 10 s |
| Snap window | 25 frames | Decoded changes are already within a second of the touch |

Smart Tagging codes are read as follows. `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`,
`BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `LONG BALL`,
`ATTACKING TRANSITION` and `SET PIECES` count for the tagged team. `DEFENDING IN ...` and
`DEFENSIVE TRANSITION` count for the other team. `N/A` tags carry no team. Tag team names are
matched to the metadata team names by string similarity.

## Assets

| Asset | Used |
|---|---|
| `tactical/BAR-ATH_tactical_atd.txt` | Yes: player and ball positions |
| `tactical/BAR-ATH_tactical_atd.xml` | Yes: team of each slot, team names |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | Yes: phase team, set-piece lead-in, kick-off |
| `tactical/BAR-ATH_tactical.mp4`, homography | No |
| `broadcast/*` | No |
| Public 5-minute ground truth | Not used to build, tune or select anything |

## Training data

The learned emissions are trained on the three public Metrica Sports sample games
(https://github.com/metrica-sports/sample-data): other matches, with tracking and event data
produced by Metrica Sports, about 428,000 frames in all. Labels for them are derived by rule from
their tracking and events, as described below; no labels were written by hand. No labelled data
from the challenge match is used for anything.

For the rule version, public sample game 3 from Metrica Sports (https://github.com/metrica-sports/sample-data, another
match, tracking and event data produced by Metrica Sports) was used to check thresholds, through
the statistics quoted in the parameter table, and to test that the pipeline runs. For this,
reference labels were derived by rule in `possession/sample.py`: dead ball where the sample
tracking has no ball, otherwise the team of the most recent `PASS`, `CARRY`, `RECOVERY`,
`SET PIECE` or `SHOT` event. No labels were written by hand.

On sample game 3, against those reference labels, the pipeline scores spellF1 0.55 and macroF1
0.84 without set-piece information, and 0.61 and 0.87 when the sample's set-piece events stand in
for Smart Tagging restarts. The sample tracking is cleaner than the match tracking (22 identified
players, ball present whenever play is live), so these figures are not an estimate for the match.

## Learned emissions (the submitted version)

The submitted file comes from this version, the default of `make_submission.py`. A classifier
replaces the hand-set evidence of the rule version. Decoding, boundary snapping and the Smart
Tagging terms (phase team, set-piece lead-in, kick-off) are unchanged.

- **Training data.** The three public Metrica sample games (other matches; tracking and events
  produced by Metrica Sports), about 428,000 frames. Labels are derived by rule
  (`possession/sample.py`): dead ball where the tracking has no ball, otherwise the team of the
  most recent `PASS`, `CARRY`, `RECOVERY`, `SET PIECE` or `SHOT` event. Nothing is labelled by
  hand and nothing from the challenge match is used.
- **Noise model** (`possession/noise.py`). The sample tracking is degraded to resemble the match
  tracking. During stoppages the ball is given a path to the restart spot, sometimes well beyond
  the line, and moved about on the way. The whole ball track is smoothed (Gaussian, sigma 5.5
  frames) with a slow wander added, because the match ball track has no sharp kicks. The ball is
  removed in blocks of at least 5 s, more often during stoppages. Players are cut into tracklets
  dealt into 46 anonymous slots, with occasional collapses in which many players are lost at
  once. Every setting was fitted to whole-match summary statistics of the match tracking printed
  by `diagnostics.py` (ball velocity-change percentiles at two time scales, share of ball
  positions off the pitch and at rest, lengths of ball-missing runs, ball visibility around
  set-piece tags, tracked players per frame, track lifetimes). No labels are involved and no
  setting was chosen by validation score.
- **Features.** The Phase 1 features plus: ball velocity change over 6 and 12 frames; frames
  since and until the ball was seen; nearest player distances at the last and next frame with a
  ball; frames since and until each team's touches; touch counts over 4 s; rolling means over 1,
  2 and 6 s; player speed before and after.
- **Touches.** The Phase 1 detector looks for a velocity change over 3 frames and finds 32
  touches in the match, because the match ball track spreads a kick over about a second. The
  learned version measures the change over 12 frames either side (threshold 8 m/s, player within
  3 m).
- **Set-piece tags.** Player speed and ball visibility averaged around the 59 set-piece tags show
  play resuming about 5 s after the start of a tag, so the learned version counts the lead-in
  up to that point as stoppage evidence.
- **Model.** LightGBM, three classes, 300 rounds, 31 leaves, learning rate 0.05, fixed seed.
  Each frame is also used mirrored (teams and labels swapped). The transition matrix is counted
  from the sample labels.
- **Emissions.** `0.5 * (log p(state | features) - log prior(state))`. The factor 0.5 accounts
  for neighbouring frames carrying the same evidence. Leave-one-game-out scores for 0.2, 0.3, 0.5
  and 1.0 are 0.542, 0.546, 0.555 and 0.559, all within noise; the middle value was kept.
- **Validation.** Leave one game out: train on two noised games, score the third with the
  official metric against its rule-derived labels.

| | Game 1 | Game 2 | Game 3 | Mean spellF1 | Mean macroF1 |
|---|---|---|---|---|---|
| Rules | 0.493 | 0.492 | 0.446 | 0.477 | 0.780 |
| Learned | 0.563 | 0.578 | 0.522 | 0.555 | 0.856 |

The same comparison repeated with two other draws of the noise model (every game re-noised,
models retrained):

| Noise seed | Rules, mean spellF1 | Learned, mean spellF1 |
|---|---|---|
| 2027 | 0.477 | 0.555 |
| 11 | 0.489 | 0.574 |
| 500 | 0.460 | 0.540 |

The learned version scores higher on every held-out game under every seed, by 0.03 to 0.13. This
comparison is what the choice of the submitted version rests on.

These scores are optimistic for the match. The held-out game is degraded by the same noise model
the classifier was trained on, and that model reproduces summary statistics of the match
tracking, not the tracking itself. The most used features are player speed after the frame and
the nearest player distances over 2 s. An earlier noise model, built from published figures
alone, gave 0.471 and 0.640; it was replaced after the match statistics showed it was wrong
about the ball track.

## Limitations

- Stoppages without a tagged set piece are found only from the ball leaving the pitch, the ball
  at rest or players slowing down. Smart Tagging lists 59 set pieces, fewer than a match has
  restarts, so some stoppages are missed or start late.
- The lead-in lengths are fixed per kind of restart. A stoppage longer than the lead-in is
  extended only if the tracking shows it.
- With the ball untracked for a long stretch, a change of possession is seen only through the
  Smart Tagging phase and collective movement, and is timed loosely.
- The reading of the Smart Tagging team tag (acting team) is an assumption from the code names.
  `diagnostics.py` prints how often attacking and defending tags of opposite teams overlap.
- Slots are tracklets. A duplicated or swapped track near the ball can credit a touch to the
  wrong team.
- The rule version's weights are set by hand, not fitted.
- The learned version is trained on other matches made to look like this one. The noise model
  reproduces summary statistics of the match tracking, not its actual errors, so the classifier
  may rely on something that does not carry over. No test data exist for this match to check it.
- Reference labels for the sample games are derived by rule from events, which is not the same
  definition as the challenge ground truth (derived from corrected tracking).

## Reproduction

See [README.md](README.md). In short, with Python 3.12, the pinned requirements, the challenge
repository next to this one and `MTAH_DATA_DIR` set:

```
python make_submission.py
```

writes `Second-Ball.csv` from the committed model in `models/`; no training is needed. Prediction
has no random component. `python make_submission.py --rules` writes the rule version.

To rebuild the model from the sample games, `python train.py`. Training uses a fixed seed and
LightGBM's deterministic mode, and gives byte-identical model files on repeated runs.
