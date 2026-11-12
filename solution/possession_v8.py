"""Possession v8: v7 plus "how many players of each team are around the ball".

    python solution/train_v6.py --version 8          # learns solution/v8_weights.json
    python solution/possession_v8.py                 # writes evaluation/results/possession_v8.csv

For every frame where the ball's position is known (seen, or bridged over a short gap), v8 counts
the players of each team within 3 m and within 8 m of it, and how close each team's SECOND
player is -- a team with two players around the ball is usually the one keeping it, or the one
pressing to win it back. Everything else is v7: the same learned classifier and decoder, trained
on external matches only.

Leave-one-match-out on nine external matches (2 Metrica, 7 IDSSE Bundesliga): v7 0.610 / 0.538
(Metrica / IDSSE) -> v8 0.613 / 0.543, better on 7 of the 9 matches and level (within 0.001) on
the other two. On the first five matches alone the gain was +0.003 and not adopted; with more
training data it became consistent.
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
import possession_v6 as v6  # noqa: E402
import possession_v7 as v7  # noqa: E402
from possession_v1 import H, A, GAME, GAMES, require  # noqa: E402

WEIGHTS_PATH = os.path.join(HERE, "v8_weights.json")
PARAMS = dict(v7.PARAMS)
PARAMS.update({"crowd_features": True})


def read_crowd(game=GAME, data_dir=None, p=PARAMS):
    """(n, 6) crowd counts around the (short-gap-bridged) ball, from the raw ATD. Cached."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    fill = int(round(p["fill_max_s"] * GAMES[game]["fps"]))
    cache = os.path.join(v1.CACHE_DIR, "{}_crowd_v8_fill{}.npz".format(game, fill))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        return np.load(cache)["crowd"]

    _, ball_filled, _ = v3.read_tracking(game, data_dir, p)
    length, width = GAMES[game]["pitch"]
    team = v1.team_of_each_track(meta_path)
    n = len(ball_filled)
    crowd = np.zeros((n, 6))
    crowd[:, 4:] = 30.0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3 or not parts[0].strip().isdigit():
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n or np.isnan(ball_filled[frame, 0]):
                continue
            bx, by = ball_filled[frame]
            xy = np.array(parts[1].replace(";", ",").split(","), dtype=float).reshape(-1, 3)[:, :2]
            dist = np.hypot((xy[:, 0] - bx) * length, (xy[:, 1] - by) * width)
            side = team[:len(dist)]
            crowd[frame] = v6.crowd_from_distances(dist[side == H][None, :],
                                                   dist[side == A][None, :])[0]
    os.makedirs(v1.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, crowd=crowd)
    return crowd


def predict(game=GAME, data_dir=None, p=PARAMS):
    model = v6.load_model(WEIGHTS_PATH)
    fps = GAMES[game]["fps"]
    ball, ball_filled, near = v3.read_tracking(game, data_dir, p)
    feats, _ = v6.features(ball, ball_filled, near, v2.read_motion(game, data_dir),
                           v4.read_directions(game, data_dir), GAMES[game]["pitch"], fps, p,
                           crowd=read_crowd(game, data_dir, p))
    emit = v6.learned_log_evidence(feats, model, p)
    return v1.predict(game, data_dir, p, tracking_log_emit=emit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v8.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v8  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
