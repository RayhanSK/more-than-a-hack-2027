"""Possession v7: v6 plus "touch" clues, learned the same way from external matches only.

    python solution/train_v6.py --version 7          # learns solution/v7_weights.json
    python solution/possession_v7.py                 # writes evaluation/results/possession_v7.csv

A touch is a sudden change in the ball's velocity -- a pass, a trap, a tackle, a header -- with a
player close by; its team is whichever is nearer. Possession changes happen at touches, so v7
gives the classifier the touch itself, which team made the last touch before each frame and the
next touch after it, and how long ago / how soon (the match is recorded, so looking ahead is
fair). Everything else is v6: same classifier, same decoder, same external training data.

Leave-one-match-out on the external matches (train on four, score the fifth): v6 0.527 -> v7
0.601 on Metrica, 0.419 -> 0.515 on IDSSE, better on every one of the five matches.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v6 as v6  # noqa: E402

WEIGHTS_PATH = os.path.join(HERE, "v7_weights.json")
PARAMS = dict(v6.PARAMS)
PARAMS.update({"touch_features": True})


def predict(game=v1.GAME, data_dir=None, p=PARAMS):
    return v6.predict(game, data_dir, p, model=v6.load_model(WEIGHTS_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v7.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v7  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
