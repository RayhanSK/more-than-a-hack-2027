"""Possession v9: v7's clues, combined by a small neural network instead of a linear model.

    python solution/train_v6.py --version 9          # learns solution/v9_weights.json
    python solution/possession_v9.py                 # writes evaluation/results/possession_v9.csv

A linear model can only add clues up; a network with one hidden layer (32 tanh units) can learn
combinations -- "close AND slow" means control, "close AND fast" means a ball flying past. It is
trained the same way as v6-v8 (external matches only, home/away-swap augmentation, full-batch
Adam from a fixed seed, so it is deterministic) and feeds the same Viterbi decoder.

On five external matches it did not beat the linear model on the Bundesliga ones. On all nine
(leave-one-match-out: train on eight, score the ninth) it does: Metrica 0.626, IDSSE 0.551,
against v8's 0.613 / 0.543, better on 8 of the 9 matches (the ninth within 0.001). It also
needs no extra pass over the tracking, unlike v8's crowd counts.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v6 as v6  # noqa: E402
import possession_v7 as v7  # noqa: E402

WEIGHTS_PATH = os.path.join(HERE, "v9_weights.json")
PARAMS = dict(v7.PARAMS)
PARAMS.update({"classifier": "mlp"})


def predict(game=v1.GAME, data_dir=None, p=PARAMS):
    return v6.predict(game, data_dir, p, model=v6.load_model(WEIGHTS_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v9.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v9  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
