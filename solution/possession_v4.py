"""Possession v4: v3 plus a "whose half is it in" prior for unclear moments.

    python solution/possession_v4.py                 # writes evaluation/results/possession_v4.csv

The idea comes from playing football: when it is not clear who has the ball -- a loose pass, a
shaky first touch, a clearance in the air -- the team defending that half is more likely to end
up with it, because it has more players there. A loose ball near Barca's goal usually stays with
Barca; a clearance from Athletic's box into Barca's half usually goes to Barca.

So on every UNCLEAR frame -- the ball is flying, or nobody is within unclear_m of it -- the
decoder gets a nudge towards the team whose own half the ball is in, growing from nothing at the
halfway line to territory_logodds at the goal line. Clear frames get no nudge: there the players
around the ball already say who has it. (Applying the nudge on every frame was tried on the
external matches and made things slightly worse.) And because the
decoder reads the whole half, an "amazing pass" that lands with the other team is still handled:
once that player controls the ball, the control vote outweighs the nudge.

Which way each team attacks is read from the tracking itself: in each half, the team whose
players are on average nearer x = 0 is defending that end.

Checked on external matches only (solution/external_eval.py): spellF1 0.444 (v3) -> 0.456, better
in both tracking scenarios. unclear_m and territory_logodds were picked there from a small grid
(3/5/8 m x odds 2..100).
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
from possession_v1 import H, A, D, GAME, GAMES, require  # noqa: E402

PARAMS = dict(v3.PARAMS)
PARAMS.update({
    # A frame is unclear when nobody is within unclear_m of the ball or the ball is flying.
    "unclear_m": 5.0,
    # Log-odds towards the defending team when the ball is on its goal line (0 at halfway).
    "territory_logodds": np.log(20.0),
})


def territory_log_evidence(ball_filled, near_filled, home_defends_left, pitch, fps, p=PARAMS):
    """(n, 3) log bonuses towards the team whose own half the ball is in, on unclear frames.

    home_defends_left: (n,) bool, True where the home team defends the x = 0 end."""
    x = ball_filled[:, 0]
    closest = np.fmin(near_filled[:, H], near_filled[:, A])
    flying = v3.ball_speed(ball_filled, pitch, fps) > p["fly_mps"]
    unclear = ~np.isnan(x) & (np.isnan(closest) | (closest > p["unclear_m"]) | flying)
    depth = np.clip(np.where(home_defends_left, 0.5 - x, x - 0.5), -0.5, 0.5) * 2  # +1 home goal
    logit = np.where(unclear, depth * p["territory_logodds"], 0.0) * p["evidence_weight"]
    out = np.zeros((len(x), 3))
    out[:, H] = logit / 2
    out[:, A] = -logit / 2
    return out


def log_evidence(ball, ball_filled, near_filled, player_speed, home_defends_left, pitch, fps,
                 p=PARAMS):
    return (v3.log_evidence(ball, ball_filled, near_filled, player_speed, pitch, fps, p)
            + territory_log_evidence(ball_filled, near_filled, home_defends_left, pitch, fps, p))


def read_directions(game=GAME, data_dir=None):
    """(n,) bool per video frame: does the home team defend the x = 0 end in that half?"""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(v1.CACHE_DIR, "{}_directions_v4.npz".format(game))
    n = v1.source_frames("tactical", game)
    blocks = v1.half_video_blocks(game)
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        left = np.load(cache)["home_left"]
    else:
        team = v1.team_of_each_track(meta_path)
        total = np.zeros((len(blocks), 2))
        count = np.zeros((len(blocks), 2))
        with open(raw_path) as handle:
            for i, line in enumerate(handle):
                if i % 25:  # one frame a second is plenty for an average
                    continue
                parts = line.rstrip("\n").split(":")
                if len(parts) != 3 or not parts[0].strip().isdigit():
                    continue
                frame = int(parts[0]) - 1
                for b, (_, first, last) in enumerate(blocks):
                    if first <= frame <= last:
                        x = np.array(parts[1].replace(";", ",").split(","),
                                     dtype=float).reshape(-1, 3)[:, 0]
                        for side in (H, A):
                            xs = x[team[:len(x)] == side]
                            xs = xs[~np.isnan(xs)]
                            total[b, side] += xs.sum()
                            count[b, side] += len(xs)
        mean = total / np.maximum(count, 1)
        left = mean[:, H] < mean[:, A]
        os.makedirs(v1.CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, home_left=left)
    out = np.zeros(n, dtype=bool)
    for b, (_, first, last) in enumerate(blocks):
        out[first:last + 1] = left[b]
    return out


def predict(game=GAME, data_dir=None, p=PARAMS):
    fps = GAMES[game]["fps"]
    ball, ball_filled, near = v3.read_tracking(game, data_dir, p)
    emit = log_evidence(ball, ball_filled, near, v2.read_motion(game, data_dir),
                        read_directions(game, data_dir), GAMES[game]["pitch"], fps, p)
    return v1.predict(game, data_dir, p, tracking_log_emit=emit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v4.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v4  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
