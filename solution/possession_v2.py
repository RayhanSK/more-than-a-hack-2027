"""Possession v2: v1 plus dead-ball evidence from ball gaps and player movement.

    python solution/possession_v2.py                 # writes evaluation/results/possession_v2.csv

v1 treated a missing ball as "no evidence", so the decoder carried the last team through every
stoppage and dead balls were badly under-predicted. v2 adds two dead-ball signals:

  * Ball gap length. In live play the tracker loses the ball for a second or two (occlusions,
    balls in the air). Through a stoppage it is gone much longer: off the pitch, in a player's
    hands, behind a crowd. The longer the gap a frame sits in, the more likely it is dead.
  * Player movement. During a stoppage players walk or stand; during play they run. Mean player
    speed is computed from the positions themselves (metres per second between consecutive
    frames), so it does not depend on the unit of the file's own speed column.

Everything else -- tracking evidence, Smart Tagging priors, Viterbi decoding -- is v1's. As in v1,
every number in PARAMS is a football-informed default, not fitted to this match. They were then
checked on external matches only (solution/external_eval.py, Metrica's public sample games):
v1 0.327 -> v2 0.416 spellF1 there. An 81-setting grid search found nothing clearly better (best
0.422, within noise and driven by the scenario where ball gaps are least like ours), so the
defaults stand.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
from possession_v1 import H, A, D, GAME, GAMES, require, source_frames  # noqa: E402

PARAMS = dict(v1.PARAMS)
PARAMS.update({
    # Ball gaps: a gap of gap_mid_s is even odds dead/alive; gap_scale_s sets how fast it turns.
    "gap_mid_s": 4.0,
    "gap_scale_s": 1.5,
    # Mean player speed: below slow_mps is evidence of a stoppage, above it of play.
    # Jogging is ~2 m/s; players at a stoppage mostly walk (~1 m/s).
    "slow_mps": 1.5,
    "speed_scale_mps": 0.4,
    # How much these two dead-ball signals count per frame, on the same footing as the
    # tracking evidence (they are just as correlated from frame to frame).
    "dead_evidence_weight": 0.1,
})


def read_motion(game=GAME, data_dir=None):
    """Per video frame: mean player speed in m/s, from positions in consecutive frames.

    NaN where fewer than 5 players are tracked in both frames. Cached under data/cache/.
    """
    raw_path = require(["atd_raw"], game, data_dir)[0]
    cache = os.path.join(v1.CACHE_DIR, "{}_motion_v2.npz".format(game))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        return np.load(cache)["speed"]

    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    n = source_frames("tactical", game)
    speed = np.full(n, np.nan)
    prev_frame, prev_xy = None, None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            xy = np.array(parts[1].replace(";", ",").split(","), dtype=float).reshape(-1, 3)[:, :2]
            if prev_frame == frame - 1 and len(prev_xy) == len(xy):
                step = np.hypot((xy[:, 0] - prev_xy[:, 0]) * length,
                                (xy[:, 1] - prev_xy[:, 1]) * width) * fps
                step = step[~np.isnan(step)]
                if len(step) >= 5:
                    # Median, so one track that jumps when the tracker swaps identities does not
                    # make a whole team look like it is sprinting.
                    speed[frame] = np.median(step)
            prev_frame, prev_xy = frame, xy

    os.makedirs(v1.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, speed=speed)
    return speed


def gap_lengths(seen):
    """For each frame, the length in frames of the run of missing-ball frames it is in (0 if the
    ball is seen)."""
    n = len(seen)
    out = np.zeros(n)
    missing = (~seen).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def smooth_mean(values, window):
    """Centred moving average ignoring NaN, so a single jittery frame does not decide."""
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    kernel = np.ones(2 * window + 1)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def dead_log_evidence(ball, speed, fps, p=PARAMS):
    """(n, 3) log-likelihoods from the two dead-ball signals: a log-odds for D, split as +x/2 on
    D and -x/2 on both alive states."""
    seen = ~np.isnan(ball[:, 0])
    gap_s = gap_lengths(seen) / fps
    logit = np.where(seen, 0.0, (gap_s - p["gap_mid_s"]) / p["gap_scale_s"])

    slow = smooth_mean(speed, fps // 2)
    logit = logit + np.where(np.isnan(slow), 0.0, (p["slow_mps"] - slow) / p["speed_scale_mps"])
    logit = np.clip(logit, -10, 10) * p["dead_evidence_weight"]

    out = np.zeros((len(ball), 3))
    out[:, D] = logit / 2
    out[:, H] = -logit / 2
    out[:, A] = -logit / 2
    return out


def predict(game=GAME, data_dir=None, p=PARAMS):
    """v1's prediction with the dead-ball evidence added before decoding."""
    fps = GAMES[game]["fps"]
    ball, _ = v1.read_tracking(game, data_dir)
    extra = dead_log_evidence(ball, read_motion(game, data_dir), fps, p)
    return v1.predict(game, data_dir, p, extra_log_emit=extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v2.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v2  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
