"""Train v6's classifier on external matches only, and test it fairly.

    python solution/train_v6.py            # leave-one-match-out test, then train on all, save
    python solution/train_v6.py --no-test  # just train on all and save solution/v6_weights.json
    python solution/train_v6.py --version 7   # the same for v7 (touch clues), v7_weights.json
    python solution/train_v6.py --version 8   # v8 (touch + crowd clues), v8_weights.json

Leave-one-match-out: for each external match, train on the others and score the one left out,
next to v5 on the same match. That is the honest estimate of how v6 does on a match it has never
seen, which is what our competition match will be. Training is deterministic (fixed degradation
seeds, full-batch gradient descent), so the weights file is reproducible.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_eval as e  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v3 as v3  # noqa: E402
import possession_v5 as v5  # noqa: E402
import possession_v6 as v6  # noqa: E402
import possession_v7 as v7  # noqa: E402
import possession_v8 as v8  # noqa: E402

TRAIN_SEED = 100   # degradation seed for training data, distinct from the scoring seeds
# (p["train_seeds"], if set, lists several: more variations of where the ball goes missing)
SUBSAMPLE = 5      # every 5th frame is plenty; neighbours are nearly identical


def game_features(g, ball, p):
    period = g["period"]
    blocks = [(i[0], i[-1]) for i in (np.flatnonzero(period == h) for h in np.unique(period))]
    filled = v3.fill_short_gaps(ball, int(round(p["fill_max_s"] * e.FPS)), blocks)
    near = e.nearest(g["home"], g["away"], filled)
    crowd = None
    if p.get("crowd_features"):
        length, width = e.PITCH
        dist = [np.hypot((pl[:, :, 0] - filled[:, None, 0]) * length,
                         (pl[:, :, 1] - filled[:, None, 1]) * width) for pl in (g["home"], g["away"])]
        crowd = v6.crowd_from_distances(*dist)
    return v6.features(ball, filled, near, g["speed"], g["home_left"], e.PITCH, e.FPS, p, crowd)


def training_set(games, p):
    X, y = [], []
    for g in games:
        for scenario in g["scenarios"]:
            base = g["ball"] if scenario == "raw" else e.fill_gaps(g["ball"], g["period"])
            for seed in p.get("train_seeds", (TRAIN_SEED,)):
                feats, signed = game_features(g, e.degrade(base, seed), p)
                X.append(feats[::SUBSAMPLE])
                y.append(g["gt"][::SUBSAMPLE])
    return np.vstack(X), np.concatenate(y), signed


def make_decode(model):
    def decode(g, ball, p, version):
        if version != 6:
            return ORIGINAL_DECODE(g, ball, p, version)
        feats, _ = game_features(g, ball, p)
        emit = v6.learned_log_evidence(feats, model, p)
        out = np.empty(len(ball), dtype=np.int8)
        log_trans = v1.log_transitions(p)
        for half in np.unique(g["period"]):
            idx = np.flatnonzero(g["period"] == half)
            out[idx] = v1.viterbi(emit[idx], log_trans)
        return out
    return decode


ORIGINAL_DECODE = e.decode


def train(games, p=v6.PARAMS):
    X, y, signed = training_set(games, p)
    if p.get("classifier") == "mlp":
        return v6.fit_mlp(X, y, signed, p)
    return v6.fit(X, y, signed, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--version", type=int, choices=(6, 7, 8), default=6)
    args = parser.parse_args()
    module = {6: v6, 7: v7, 8: v8}[args.version]
    params = module.PARAMS

    games = e.load_all()
    print("external matches:", [g["name"] for g in games])
    if not args.no_test:
        rows = []
        for held in games:
            model = train([g for g in games if g is not held], params)
            e.decode = make_decode(model)
            s6 = e.score([held], params, 6)[0]
            e.decode = ORIGINAL_DECODE
            s5 = e.score([held], v5.PARAMS, 5)[0]
            rows.append((held["name"], s5, s6))
            print("{:<10} v5 {:.4f}  v{} {:.4f}".format(rows[-1][0], rows[-1][1], args.version,
                                                     rows[-1][2]), flush=True)
        met = [r for r in rows if r[0].startswith("metrica")]
        ids = [r for r in rows if not r[0].startswith("metrica")]
        for label, sub in (("metrica", met), ("idsse", ids)):
            if sub:
                print("{:<10} v5 {:.4f}  v{} {:.4f}".format(
                    label, np.mean([r[1] for r in sub]), args.version,
                    np.mean([r[2] for r in sub])))

    model = train(games, params)
    model["trained_on"] = [g["name"] for g in games]
    with open(module.WEIGHTS_PATH, "w") as handle:
        json.dump(model, handle, indent=1)
    print("-> {}".format(os.path.relpath(module.WEIGHTS_PATH, v1.ROOT)))
