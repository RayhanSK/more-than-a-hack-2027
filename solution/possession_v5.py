"""Possession v5: v4 plus "the ball is sitting still" as evidence of a dead ball.

    python solution/possession_v5.py                 # writes evaluation/results/possession_v5.csv

In live play the ball is almost never still for long: even a player shielding it keeps it moving.
At a stoppage it is placed for a free kick, a goal kick or a corner, or held for a throw-in, and
it barely moves. So wherever the ball is tracked, a ball whose speed (averaged over about a second)
is below still_mps is evidence of a dead ball, growing as it gets slower.

Checked on external matches only (solution/external_eval.py). On the Bundesliga matches, where
the ball is tracked through stoppages and the official DFL feed says when play is live, spellF1
went from 0.263 (v4) to 0.337. On Metrica's games, whose ball disappears at stoppages, it changes
nothing, as expected. still_mps, still_scale_mps and still_weight were picked there from a small
grid.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import possession_v3 as v3  # noqa: E402
import possession_v4 as v4  # noqa: E402
from possession_v1 import H, A, D, GAME, GAMES  # noqa: E402

PARAMS = dict(v4.PARAMS)
PARAMS.update({
    # Ball slower than this (m/s, averaged over ~1 s) is evidence of a stoppage.
    "still_mps": 1.0,
    "still_scale_mps": 0.5,
    # Per-frame weight of that evidence (the tracking votes use evidence_weight = 0.1).
    "still_weight": 0.3,
})


def still_log_evidence(ball, pitch, fps, p=PARAMS):
    """(n, 3) log-likelihoods: a D log-odds where the tracked ball is (nearly) still."""
    speed = v2.smooth_mean(v3.ball_speed(ball, pitch, fps), fps // 2)
    logit = np.where(np.isnan(speed), 0.0,
                     np.clip((p["still_mps"] - speed) / p["still_scale_mps"], 0.0, 10.0))
    logit = logit * p["still_weight"]
    out = np.zeros((len(ball), 3))
    out[:, D] = logit / 2
    out[:, H] = -logit / 2
    out[:, A] = -logit / 2
    return out


def log_evidence(ball, ball_filled, near_filled, player_speed, home_defends_left, pitch, fps,
                 p=PARAMS):
    return (v4.log_evidence(ball, ball_filled, near_filled, player_speed, home_defends_left,
                            pitch, fps, p)
            + still_log_evidence(ball, pitch, fps, p))


def predict(game=GAME, data_dir=None, p=PARAMS):
    fps = GAMES[game]["fps"]
    ball, ball_filled, near = v3.read_tracking(game, data_dir, p)
    emit = log_evidence(ball, ball_filled, near, v2.read_motion(game, data_dir),
                        v4.read_directions(game, data_dir), GAMES[game]["pitch"], fps, p)
    return v1.predict(game, data_dir, p, tracking_log_emit=emit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v5.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v5  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
