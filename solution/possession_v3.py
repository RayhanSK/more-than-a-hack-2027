"""Possession v3: v2 plus better use of the ball itself.

    python solution/possession_v3.py                 # writes evaluation/results/possession_v3.csv

v1 and v2 treat every frame where the ball is seen alike: whichever team is nearest gets a vote of
the same size. v3 changes three things about that vote:

  * Short gaps are bridged. When the ball disappears for under fill_max_s, its path is filled in
    by a straight line, and the nearest players to that path still vote, at a reduced weight.
    Long gaps are left alone, because they are v2's dead-ball signal.
  * A flying ball counts for little. When the ball moves faster than fly_mps (a pass, a clearance,
    a shot), the nearest player is usually just someone it is flying past.
  * Close control counts for more. When a player is within control_m of a slow ball, that team
    almost certainly has it.

Ball speed is computed from the positions themselves, in metres per second. The dead-ball signals
(gap length, player movement), Smart Tagging priors and Viterbi decoding are v2's. Every number in
PARAMS is a football-informed default, checked on external matches only
(solution/external_eval.py), never on this match. There, spellF1 went from v2's 0.416 to 0.444.
Nearly all of the gain comes from close control (0.411 without it); bridging gaps and
discounting flying balls were roughly neutral, and nudging any setting moved the score by less
than 0.003, so the defaults stand.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
from possession_v1 import H, A, GAME, GAMES, require  # noqa: E402

PARAMS = dict(v2.PARAMS)
PARAMS.update({
    # Ball gaps up to this long are bridged by a straight line.
    "fill_max_s": 1.0,
    # Weight of a vote on a bridged frame, relative to a seen one.
    "filled_factor": 0.5,
    # Ball faster than this (m/s) is in flight; its votes are scaled by fly_factor.
    "fly_mps": 12.0,
    "fly_factor": 0.2,
    # A player within control_m of a ball slower than control_mps has it; votes scaled up.
    "control_m": 1.0,
    "control_mps": 5.0,
    "control_factor": 3.0,
})


# -- ball handling --------------------------------------------------------------------------

def fill_short_gaps(ball, max_frames, blocks=None):
    """Copy of ball with runs of at most max_frames missing frames filled by straight lines.
    Gaps at the start or end of a block are left missing. blocks: list of (first, last) inclusive
    index ranges that must not be bridged across (the halves); default is the whole array."""
    out = ball.copy()
    if blocks is None:
        blocks = [(0, len(ball) - 1)]
    for first, last in blocks:
        seen = ~np.isnan(ball[first:last + 1, 0])
        edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start > max_frames or start == 0 or stop == len(seen):
                continue
            a, b = first + start - 1, first + stop  # last seen before, first seen after
            t = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = ball[a] + t[:, None] * (ball[b] - ball[a])
    return out


def ball_speed(ball, pitch, fps):
    """Ball speed in m/s, averaged over a short window to soften tracking jitter."""
    length, width = pitch
    step = np.hypot(np.diff(ball[:, 0]) * length, np.diff(ball[:, 1]) * width) * fps
    speed = np.concatenate([[np.nan], step])
    return v2.smooth_mean(speed, 2)


def vote_weights(seen, filled, near, speed, p=PARAMS):
    """Per-frame multiplier on the tracking vote."""
    w = np.where(seen, 1.0, np.where(filled, p["filled_factor"], 0.0))
    closest = np.fmin(near[:, H], near[:, A])
    fast = speed > p["fly_mps"]
    control = (closest < p["control_m"]) & (speed < p["control_mps"])
    w = np.where(fast, w * p["fly_factor"], w)
    w = np.where(control & ~fast, w * p["control_factor"], w)
    return w


def log_evidence(ball, ball_filled, near_filled, player_speed, pitch, fps, p=PARAMS):
    """(n, 3) log-likelihoods from the tracking: v3's weighted votes plus v2's dead-ball signals.

    ball is the ball as tracked (gaps intact, for v2's gap signal); ball_filled has short gaps
    bridged, and near_filled is the nearest-player distance to it."""
    seen = ~np.isnan(ball[:, 0])
    filled = ~seen & ~np.isnan(ball_filled[:, 0])
    w = vote_weights(seen, filled, near_filled, ball_speed(ball_filled, pitch, fps), p)
    votes = v1.tracking_log_evidence(ball_filled, near_filled, p)  # already * evidence_weight
    return votes * w[:, None] + v2.dead_log_evidence(ball, player_speed, fps, p)


# -- reading this match ---------------------------------------------------------------------

def read_tracking(game=GAME, data_dir=None, p=PARAMS):
    """Ball as tracked, ball with short gaps bridged, and nearest home/away distance to the
    bridged ball. Cached under data/cache/ per fill_max_s."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    fill = int(round(p["fill_max_s"] * GAMES[game]["fps"]))
    cache = os.path.join(v1.CACHE_DIR, "{}_tracking_v3_fill{}.npz".format(game, fill))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["ball"], z["ball_filled"], z["near"]

    ball, near = v1.read_tracking(game, data_dir)
    blocks = [(first, last) for _, first, last in v1.half_video_blocks(game)]
    ball_filled = fill_short_gaps(ball, fill, blocks)
    near = near.copy()
    todo = np.zeros(len(ball), dtype=bool)
    todo[np.isnan(ball[:, 0]) & ~np.isnan(ball_filled[:, 0])] = True

    length, width = GAMES[game]["pitch"]
    team = v1.team_of_each_track(meta_path)
    with open(raw_path) as handle:
        for line in handle:
            head = line.split(":", 1)[0]
            if not head.strip().isdigit():
                continue
            frame = int(head) - 1
            if not (0 <= frame < len(todo) and todo[frame]):
                continue
            parts = line.rstrip("\n").split(":")
            bx, by = ball_filled[frame]
            xy = np.array(parts[1].replace(";", ",").split(","), dtype=float).reshape(-1, 3)[:, :2]
            dist = np.hypot((xy[:, 0] - bx) * length, (xy[:, 1] - by) * width)
            for side in (H, A):
                d = dist[team[:len(dist)] == side]
                d = d[~np.isnan(d)]
                if len(d):
                    near[frame, side] = d.min()

    os.makedirs(v1.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, ball=ball, ball_filled=ball_filled, near=near)
    return ball, ball_filled, near


def predict(game=GAME, data_dir=None, p=PARAMS):
    """Labels for every video frame, decoded like v1 with v3's evidence."""
    fps = GAMES[game]["fps"]
    ball, ball_filled, near = read_tracking(game, data_dir, p)
    emit = log_evidence(ball, ball_filled, near, v2.read_motion(game, data_dir),
                        GAMES[game]["pitch"], fps, p)
    return v1.predict(game, data_dir, p, tracking_log_emit=emit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(v1.RESULTS_DIR, "possession_v3.csv"))
    args = parser.parse_args()

    labels = predict()
    path = v1.write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v3  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))
