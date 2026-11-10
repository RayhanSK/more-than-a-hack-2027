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
    if p.get("touch_features"):
        cols += touch_features(ball_filled, margin, closest, known, pitch, fps)
    feats = np.column_stack([np.asarray(c, dtype=float) for c, _ in cols])
    signed = np.array([s for _, s in cols])
    return feats, signed


def _carry(values, valid, reverse=False):
    """Forward-fill (or backward-fill) values from frames where valid; 0 before the first one.
    Also returns frames since (or until) that frame."""
    n = len(values)
    order = np.arange(n)[::-1] if reverse else np.arange(n)
    idx = np.where(valid[order], order, -1)
    if reverse:
        idx = np.where(idx < 0, n, idx)
        last = np.minimum.accumulate(idx)
        ok = last < n
    else:
        last = np.maximum.accumulate(idx)
        ok = last >= 0
    last, ok = last[::-1] if reverse else last, ok[::-1] if reverse else ok
    held = np.where(ok, values[np.clip(last, 0, n - 1)], 0.0)
    dist = np.where(ok, np.abs(np.arange(n) - last), 10 ** 6)
    return held, dist


def touch_features(ball_filled, margin, closest, known, pitch, fps, touch_mps=3.0, touch_m=2.0):
    """v7's extra clues: a TOUCH is a sudden change in the ball's velocity (a pass, a trap, a
    tackle) with a player close by; its team is whichever is nearer. Features: the touch itself
    (home-signed), and which team made the last touch before and the next touch after each frame,
    with how long ago / how soon (the decoder can use the future: the match is recorded)."""
    length, width = pitch
    xy = np.column_stack([ball_filled[:, 0] * length, ball_filled[:, 1] * width])
    vel = np.full_like(xy, np.nan)
    vel[1:-1] = (xy[2:] - xy[:-2]) * fps / 2
    dv = np.full(len(xy), np.nan)
    dv[2:-2] = np.hypot(*(vel[4:] - vel[:-4]).T)
    dv = np.where(np.isnan(dv), 0.0, np.minimum(dv, 30.0))
    side = np.sign(margin)
    touch = known & (dv > touch_mps) & (closest < touch_m) & (side != 0)
    last_side, since = _carry(side, touch)
    next_side, until = _carry(side, touch, reverse=True)
    since_s, until_s = np.minimum(since / fps, 30.0), np.minimum(until / fps, 30.0)
    return [
        (np.tanh(dv / 5.0) * (closest < touch_m) * side, True),
        (last_side, True), (last_side * np.exp(-since_s / 3.0), True),
        (next_side, True), (next_side * np.exp(-until_s / 3.0), True),
        ((last_side == next_side) * last_side, True),
        (np.log1p(since_s), False), (np.log1p(until_s), False),
    ]


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


def fit_mlp(X, y, signed, p=PARAMS, hidden=32, steps=600, lr=0.01, seed=0):
    """A one-hidden-layer network (tanh), trained by full-batch Adam from a fixed seed, so it is
    deterministic. Same home/away-swap augmentation as fit(). It can learn combinations of clues
    ("close AND slow") that a linear model cannot."""
    Xs = X.copy()
    Xs[:, signed] *= -1
    ys = np.where(y == H, A, np.where(y == A, H, y))
    X, y = np.vstack([X, Xs]), np.concatenate([y, ys])
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    Y = np.eye(3)[y]
    rng = np.random.default_rng(seed)
    params = [rng.normal(0, 1 / np.sqrt(Z.shape[1]), (Z.shape[1], hidden)), np.zeros(hidden),
              rng.normal(0, 1 / np.sqrt(hidden), (hidden, 3)), np.zeros(3)]
    m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]
    for t in range(1, steps + 1):
        W1, b1, W2, b2 = params
        h = np.tanh(Z @ W1 + b1)
        logits = h @ W2 + b2
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        g = (P - Y) / len(Z)
        gW2 = h.T @ g + p["l2"] * W2
        gb2 = g.sum(0)
        gh = (g @ W2.T) * (1 - h ** 2)
        gW1 = Z.T @ gh + p["l2"] * W1
        gb1 = gh.sum(0)
        for q, gq, mq, vq in zip(params, (gW1, gb1, gW2, gb2), m, v):
            mq *= 0.9
            mq += 0.1 * gq
            vq *= 0.999
            vq += 0.001 * gq ** 2
            q -= lr * (mq / (1 - 0.9 ** t)) / (np.sqrt(vq / (1 - 0.999 ** t)) + 1e-8)
    prior = np.bincount(y, minlength=3) / len(y)
    return {"kind": "mlp", "mean": mean.tolist(), "std": std.tolist(),
            "W1": params[0].tolist(), "b1": params[1].tolist(), "W2": params[2].tolist(),
            "b2": params[3].tolist(), "log_prior": np.log(prior).tolist()}


def learned_log_evidence(feats, model, p=PARAMS):
    """(n, 3) scaled log-likelihoods: log P(class | x) - log P(class), times learned_weight."""
    Zn = (feats - np.array(model["mean"])) / np.array(model["std"])
    if model.get("kind") == "mlp":
        h = np.tanh(Zn @ np.array(model["W1"]) + np.array(model["b1"]))
        logits = h @ np.array(model["W2"]) + np.array(model["b2"])
    else:
        logits = np.column_stack([Zn, np.ones(len(feats))]) @ np.array(model["W"])
    logits = logits - logits.max(1, keepdims=True)
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
