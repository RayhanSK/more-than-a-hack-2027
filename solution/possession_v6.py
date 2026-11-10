"""Possession v6: the same clues as v5, but how much each one counts is LEARNED from external
matches instead of set by hand.

    python solution/train_v6.py                      # learns solution/v6_weights.json (external only)
    python solution/possession_v6.py                 # writes evaluation/results/possession_v6.csv

v1-v5 combine their clues (who is nearer the ball, close control, ball speed, ball gaps, player
speed, whose half) with weights chosen by football reasoning and small grid searches. v6 turns
every clue into a number per frame and fits a multinomial logistic regression -- the simplest
classifier there is -- that maps them to P(H), P(A), P(D). It is trained ONLY on external matches
(Metrica's two sample games and three IDSSE Bundesliga games); the provided match's labels and
video are never used. Its probabilities become the decoder's evidence, and the Viterbi decoder
with its switch costs is unchanged, so the output is still whole possessions.

The features are "home-signed" (positive means "points to the home team"), and training also
sees every frame with home and away swapped, so the classifier cannot learn that one side simply
has the ball more often.

Leave-one-match-out on the external matches (train on four, score the fifth): v5 0.489 -> v6
0.527 on Metrica, 0.377 -> 0.419 on IDSSE, better on every one of the five matches.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import possession_v3 as v3  # noqa: E402
import possession_v4 as v4  # noqa: E402
import possession_v5 as v5  # noqa: E402
from possession_v1 import H, A, D, GAME, GAMES  # noqa: E402

WEIGHTS_PATH = os.path.join(HERE, "v6_weights.json")

PARAMS = dict(v5.PARAMS)
PARAMS.update({
    # How much one frame's learned log-probability counts (frames are far from independent).
    # Picked by leave-one-match-out on the external matches: 0.1 / 0.2 / 0.3 / 0.5 gave mean
    # spellF1 0.421 / 0.459 / 0.473 / 0.474; 0.3 is the smaller of the two equal best.
    "learned_weight": 0.3,
    # L2 penalty and steps for the logistic regression.
    "l2": 1e-3,
    "steps": 400,
})


# -- features -------------------------------------------------------------------------------

def features(ball, ball_filled, near, player_speed, home_left, pitch, fps, p=PARAMS):
    """(n, F) home-signed features and (F,) bool mask of which are signed (flip on swap)."""
    seen = ~np.isnan(ball[:, 0])
    filled = ~seen & ~np.isnan(ball_filled[:, 0])
    known = seen | filled
    dh = np.where(np.isnan(near[:, H]), 30.0, np.minimum(near[:, H], 30.0))
    da = np.where(np.isnan(near[:, A]), 30.0, np.minimum(near[:, A], 30.0))
    dh, da = np.where(known, dh, 30.0), np.where(known, da, 30.0)
    margin = np.clip(da - dh, -10, 10)
    closest = np.minimum(dh, da)
    bspeed = v3.ball_speed(ball_filled, pitch, fps)
    bspeed = np.where(np.isnan(bspeed), 0.0, np.minimum(bspeed, 30.0))
    still = v2.smooth_mean(v3.ball_speed(ball, pitch, fps), fps // 2)
    still_ok = ~np.isnan(still)
    still = np.where(still_ok, np.minimum(still, 10.0), 0.0)
    gap_s = v2.gap_lengths(seen) / fps
    ps = v2.smooth_mean(player_speed, fps // 2)
    ps_ok = ~np.isnan(ps)
    ps = np.where(ps_ok, np.minimum(ps, 8.0), 0.0)
    x = ball_filled[:, 0]
    depth = np.where(known, np.clip(np.where(home_left, 0.5 - x, x - 0.5), -0.5, 0.5) * 2, 0.0)
    m = p["outside_margin"]
    outside = known & ((ball_filled[:, 0] < -m) | (ball_filled[:, 0] > 1 + m) |
                       (ball_filled[:, 1] < -m) | (ball_filled[:, 1] > 1 + m))
    control = known & (closest < p["control_m"])
    signed_control = control * np.sign(margin)
    cols = [
        # (feature, signed?)
        (seen, False), (filled, False), (outside, False),
        (np.tanh(margin / 1.0), True), (np.tanh(margin / 3.0), True),
        (np.tanh(v2.smooth_mean(margin, fps // 2) / 2.0), True),
        (np.tanh(v2.smooth_mean(margin, 2 * fps) / 2.0), True),
        (signed_control, True),
        (np.log1p(closest), False), (control, False),
        (np.log1p(bspeed), False), (bspeed > p["fly_mps"], False),
        (still_ok, False), (np.exp(-still), False),
        (np.log1p(gap_s), False), (gap_s > 2.0, False),
        (ps_ok, False), (ps, False),
        (depth, True), (depth * (closest > p["unclear_m"]), True),
    ]
    feats = np.column_stack([np.asarray(c, dtype=float) for c, _ in cols])
    signed = np.array([s for _, s in cols])
    return feats, signed


# -- the classifier -------------------------------------------------------------------------

def fit(X, y, signed, p=PARAMS):
    """Multinomial logistic regression by full-batch gradient descent (deterministic).
    Training data is doubled with home/away swapped. Returns a dict of plain lists."""
    Xs = X.copy()
    Xs[:, signed] *= -1
    ys = np.where(y == H, A, np.where(y == A, H, y))
    X, y = np.vstack([X, Xs]), np.concatenate([y, ys])
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([(X - mean) / std, np.ones(len(X))])
    W = np.zeros((Z.shape[1], 3))
    Y = np.eye(3)[y]
    lr = 0.5
    for _ in range(p["steps"]):
        logits = Z @ W
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        grad = Z.T @ (P - Y) / len(Z) + p["l2"] * np.vstack([W[:-1], np.zeros((1, 3))])
        W -= lr * grad
    prior = np.bincount(y, minlength=3) / len(y)
    return {"mean": mean.tolist(), "std": std.tolist(), "W": W.tolist(),
            "log_prior": np.log(prior).tolist()}


def learned_log_evidence(feats, model, p=PARAMS):
    """(n, 3) scaled log-likelihoods: log P(class | x) - log P(class), times learned_weight."""
    Z = np.column_stack([(feats - np.array(model["mean"])) / np.array(model["std"]),
                         np.ones(len(feats))])
    logits = Z @ np.array(model["W"])
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return (logp - np.array(model["log_prior"])) * p["learned_weight"]


def load_model(path=WEIGHTS_PATH):
    with open(path) as handle:
        return json.load(handle)


# -- this match -----------------------------------------------------------------------------

def predict(game=GAME, data_dir=None, p=PARAMS, model=None):
    model = model or load_model()
    fps = GAMES[game]["fps"]
    ball, ball_filled, near = v3.read_tracking(game, data_dir, p)
    feats, _ = features(ball, ball_filled, near, v2.read_motion(game, data_dir),
                        v4.read_directions(game, data_dir), GAMES[game]["pitch"], fps, p)
    emit = learned_log_evidence(feats, model, p)
    return v1.predict(game, data_dir, p, tracking_log_emit=emit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v6.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v6  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
