# Possession from automatic tracking — technical document

**Team:** `<TEAM NAME>` · **Repository:** `<PUBLIC REPO URL>` · **Submission file:** `<TEAM NAME>.csv`

## 1. Summary

We label every frame of the tactical video `H`, `A`, `D` or `X` from the automatic tracking (ATD)
and the Smart Tagging. Two steps:

1. **Per-frame evidence.** Each frame gets a set of clues from the tracking — which team is nearer
   the ball, whether a player has it under close control, how fast the ball moves, whether it is
   sitting still, how long it has been missing, how fast the players move, whose half it is in,
   and which team made the last and the next *touch*. A small classifier (multinomial logistic
   regression) turns the clues into P(H), P(A), P(D). It is trained **only on external matches**.
2. **Whole possessions.** A three-state hidden Markov model with switch costs, decoded by Viterbi
   over each half, turns those noisy per-frame probabilities into whole possessions. A change of
   possession is only made when the evidence for it lasts long enough to pay its cost, which is
   what the possession-level ranking metric (spellF1) rewards.

On external matches it was never trained on, the final model (v7) scores spellF1 **0.616** on
Metrica's two public sample games and **0.527** on three Bundesliga matches, against **0.327** and
**0.193** for our first version (section 5).

## 2. Assets used, and how

| asset | used? | how |
|---|---|---|
| ATD tracking (`atd.txt`, `atd.xml`) | yes, the main input | ball and player positions; team of each track slot from the metadata. All distances in metres using the 105.3 × 68 m pitch. |
| Smart Tagging | yes, as a weak prior | set-piece tags nudge the seconds before a restart towards `D`; phases tagged with a team (Build Up, Progression, …; Defending … for the opponent) nudge towards that team. Not used for training or tuning (no external equivalent). |
| Homography | no | not needed: the ATD is already in pitch coordinates. |
| Tactical video | no | never opened or used. |
| Public 5-minute ground truth | pipeline check only | used only to confirm the file is produced and scored correctly. Never used to choose a model, a feature or a setting. |

## 3. Training and tuning data (all external)

No label, statistic or setting comes from the provided game. Every learned weight and every tuned
setting comes from these datasets, all public:

| dataset | matches | labels | who labelled / how |
|---|---|---|---|
| **Metrica Sports sample data** (github.com/metrica-sports/sample-data) | 2 full matches | per frame, derived by our own algorithm from Metrica's event data: set piece → that team; ball out / foul / shot out → `D` until the next event; other team events → that team; challenges and cards skipped (a contested ball stays with the last team, as in the challenge rules) | Metrica (events); derivation in `solution/external_eval.py` |
| **IDSSE** — Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Sci Data 12, 195 (2025), CC BY 4.0 | 3 Bundesliga 2022/23 matches: Köln v Bayern, Bochum v Leverkusen, Düsseldorf v Nürnberg | per frame: official DFL `BallStatus` (alive/dead) and `BallPossession` (home/guest) | DFL's official tracking provider |

The IDSSE labels are produced the same way the challenge describes its own ground truth — from
(corrected) tracking that marks the ball dead and names the owning team — so we treat them as the
more trustworthy proxy, and prefer changes that help there.

**Matching the competition's tracking quality.** External tracking is cleaner than the ATD, so
every evaluation first removes the ball in random bursts (seeded) until about 40% of frames have
no ball, the share reported for the ATD. Metrica's sample also loses the ball at almost every
stoppage, which would flatter "missing ball = dead"; it is scored twice, with its own gaps and with
them interpolated away, and the two are averaged.

**Fair testing.** For the learned models, each external match is scored by a model trained on the
other four (leave-one-match-out), so every score is on a match the model has never seen, which
is the situation on the provided game.

## 4. The model, version by version

Every version is in `solution/` and can be re-scored with `python solution/external_eval.py`.

| version | adds | Metrica | IDSSE |
|---|---|---|---|
| v1 | nearest-team evidence, ball outside the pitch, Smart Tagging priors, Viterbi with switch costs | 0.327 | 0.193 |
| v2 | dead-ball evidence: long ball gaps, slow player movement | 0.416 | 0.226 |
| v3 | close control (player within 1 m of a slow ball) counts more; flying balls count less; short gaps bridged | 0.444 | 0.253 |
| v4 | on unclear frames, lean towards the team defending that half | 0.456 | 0.263 |
| v5 | a ball sitting still is evidence of a stoppage; sharper nearest-team vote | 0.489 | 0.377 |
| v6 | clue weights learned by logistic regression on the external matches | 0.527 | 0.419 |
| **v7 (submitted)** | **ball-touch clues: team of the last and next touch, and how recent** | **0.616** | **0.527** |

v1–v5 are scored on the external matches directly (their few settings were chosen there by small
grid searches); v6–v7 by leave-one-match-out.

**Clues (v7 features).** All are computed from the tracking alone. "Home-signed" features point
towards the home team when positive; training also sees every frame with home and away swapped,
so the classifier cannot learn that one side simply has the ball more.

- distance from the ball to the nearest home and away player; their difference at three time
  scales; close control (nearest player within 1 m)
- ball speed, flying ball (> 12 m/s), ball nearly still (averaged over ~1 s)
- ball seen / bridged / missing, and the length of the gap it is in; ball outside the pitch
- median player speed (players walk at stoppages)
- depth into each team's half (attack direction per half read from the players' mean position)
- **touches**: a sudden change in ball velocity (> 3 m/s) with a player within 2 m; the touching
  team is the nearer one. Features: the touch, the team of the last touch before and the next touch
  after each frame, and how long ago / how soon. Looking ahead is fair: the match is recorded and
  the decoder reads each half as a whole.

**Decoder.** Three states, H/A/D. Switch-team probability 1/250 per frame, alive→dead 1/1500,
dead→alive 1/500 (≈ 10 s possessions, a stoppage a minute, ≈ 20 s stoppages); decoded separately
for each half, `X` outside the halves. The classifier's log-probabilities minus the class priors
are used as emissions, weighted 0.5 per frame because consecutive frames are far from independent.

## 5. Possession rules we implemented

- **Loose and contested balls stay with the last team.** A single frame where the ball is between
  two players gives almost no evidence, and the switch cost keeps the current team until the other
  team's evidence lasts. This is the challenge's own rule.
- **Control, not contact.** A brief touch or deflection by the other team is not enough to switch:
  it has to be followed by that team staying nearest the ball, or by more touches. Close control
  and the last/next-touch clues carry most of this.
- **Unclear moments** (flying ball, or nobody within 5 m) lean towards the team defending that
  half — a loose ball in a team's own half usually stays with it (a rule from our own playing
  experience, kept because it helped on the external matches).
- **Restarts and stoppages.** Dead ball is detected from: the ball outside the pitch, a long
  ball gap, a ball sitting still (placed for a free kick, held for a throw-in), slow player
  movement, and the Smart Tagging set-piece windows. Who will restart is deliberately not asked;
  a stoppage is always `D`.
- **No `X` inside the halves.** Every graded frame gets `H`, `A` or `D`.

## 6. What we tried and dropped

All measured with the same leave-one-match-out test (Metrica / IDSSE, v7 = 0.616 / 0.527):

- a small neural network instead of the logistic regression: 0.638 / 0.524 — better on Metrica
  only; with the players-around-the-ball clue 0.634 / 0.530 — better on 3 of 5 matches only
- players of each team within 3 m / 8 m of the ball: +0.003, not worth an extra pass over the data
- "controlled possession" counters (touches per team over 5 s, same-team touch streaks, time on the
  ball): worse, 0.606 / 0.511
- more ball-dropout variations in training; retuning the switch costs: no change
- training on the Bundesliga matches only: worse on them than training on all five
- homography and video: not used (see section 2)

## 7. Limitations

- **Labels differ between datasets.** Each source draws "possession" and "dead" slightly
  differently (Metrica's labels are derived from events, DFL's from tracking), and neither is the
  challenge's own labelling. The model is tuned to what they share.
- **The ATD loses the ball 41% of the time** on graded frames, more at exactly the moments that
  matter (crowded boxes, balls in the air). Long gaps inside live play are where it fails most.
- **Throw-ins are tagged `N/A`** in the Smart Tagging, so the set-piece prior cannot say whose
  restart it is; this is not needed for `D`, but it is for the possession that follows.
- **Track slots are not players**, so we cannot follow one player's control over time; touch and
  control clues are per team.
- **Small external training set** (five matches). With more time: more matches, a learned
  transition model, and video-based ball detection to fill the ATD's gaps.

## 8. Reproduction

Python 3.10, numpy 2.2.6, pandas 2.3.3 (`solution/requirements.txt`). With the data bundle in
`./data` (or `MTAH_DATA_DIR` pointing at it):

```bash
pip install -r solution/requirements.txt
python solution/make_submission.py --team "<TEAM NAME>"
```

This writes `evaluation/results/<TEAM NAME>.csv`, runs the official format check and prints the
file's SHA-256. `--fresh` re-parses the raw data instead of using `data/cache/`; both give a
byte-identical file. The learned weights are in `solution/v7_weights.json`; to recreate them from
the external data, download it as described in `solution/README.md` and run
`python solution/train_v6.py --version 7`.

## 9. Use of AI assistance

`<TO CONFIRM WITH THE TEAM>` The code was written with the help of an AI coding assistant. In line
with the rules, none of the provided data (tracking, tagging, video or ground truth) was uploaded
to it: the pipeline ran on our own computer, and the assistant saw only code, error messages, label counts, format-check results and the
pipeline-check score on the public 5-minute window.
