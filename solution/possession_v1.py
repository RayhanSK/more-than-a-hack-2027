"""Possession v1: tracking evidence + Smart Tagging priors, decoded as whole possessions.

    python solution/possession_v1.py                 # writes evaluation/results/possession_v1.csv

Three steps:

  1. Evidence from the tracking. On every frame where the ball is seen, how much closer is the
     nearest home player than the nearest away player, and is anyone close to the ball at all?
     A ball outside the pitch is evidence of a dead ball. A missing ball is no evidence.
  2. Priors from the Smart Tagging. A set piece marks a restart, so the seconds just before it are
     probably dead. A phase tagged with a team (Build Up, Progression, Defending...) is weak
     evidence about which team has the ball.
  3. Decoding with Viterbi over three states (H, A, D). Every change of state has a cost, so the
     decoder only changes possession when the evidence against the current state lasts long
     enough to pay for it. That is what turns flickering frame guesses into whole possessions,
     and possessions are what spellF1 counts.

Every number in PARAMS is a football-informed default, NOT fitted to this match. The rules forbid
tuning on the provided ground truth or video; tuning happens on external matches (see the team
plan). The public five minutes are used only to check that the pipeline is wired correctly.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import load_smart_tagging  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_DIR = os.path.join(ROOT, "data", "cache")  # under data/, so gitignored with the match data

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Params -- defaults from football reasoning, to be tuned on external matches only.
PARAMS = {
    # How strongly one frame of tracking counts. Consecutive frames at 25 fps are far from
    # independent, so each is down-weighted; 0.1 means roughly 2.5 "independent looks" a second.
    "evidence_weight": 0.1,
    # Metres of "closer than the nearest opponent" that make a team ~73% likely (logistic scale).
    "margin_scale_m": 1.5,
    # Probability the ball is alive when the nearest player is within near_m / beyond far_m.
    "p_alive_near": 0.9,
    "p_alive_far": 0.6,
    "near_m": 2.0,
    "far_m": 10.0,
    # Probability the ball is alive when the tracked ball is outside the pitch lines.
    "p_alive_outside": 0.1,
    "outside_margin": 0.01,  # in normalised pitch units, to absorb tracking jitter on the line
    # Per-frame transition probabilities. Typical possessions last ~10 s, stoppages occur roughly
    # once a minute, and a stoppage lasts ~20 s.
    "p_switch_team": 1.0 / (25 * 10),
    "p_alive_to_dead": 1.0 / (25 * 60),
    "p_dead_to_alive": 1.0 / (25 * 20),
    # Smart Tagging: the set-piece tag is a ~5 s window around the restart. Frames from pre_s
    # before its start to post_s after it get a bonus towards D.
    "setpiece_pre_s": 3.0,
    "setpiece_post_s": 1.0,
    "setpiece_dead_bonus": np.log(4.0),
    # Smart Tagging: phases tagged with a team nudge towards that team (or its opponent).
    "phase_team_bonus": np.log(1.5),
}

# Codes whose Team tag names the team in possession, and codes where it names the team without
# the ball. Matched case-insensitively.
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "long ball", "ball in final third", "ball in the box", "players in the box"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


# -- step 1: tracking evidence -------------------------------------------------------------

def team_of_each_track(meta_path):
    """Array over track slots: H if the slot belongs to the first (home) team, else A."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([H if p.get("teamId") == home_id else A for p in root.iter("Player")],
                    dtype=np.int8)


def read_tracking(game=GAME, data_dir=None):
    """Per video frame: ball x, y (normalised, NaN if missing) and the distance in metres from the
    ball to the nearest home and nearest away player (NaN if no ball or no player).

    Parsing 199k lines takes a minute or two, so the result is cached under data/cache/.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(CACHE_DIR, "{}_tracking_v1.npz".format(game))
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["ball"], z["near"]

    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    ball = np.full((n, 2), np.nan)
    near = np.full((n, 2), np.nan)  # [:, H] nearest home, [:, A] nearest away

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            bx, by = (float(v) for v in parts[2].split(",")[:2])
            if bx != bx:  # NaN: no ball on this frame
                continue
            ball[frame] = bx, by
            xy = np.array(parts[1].replace(";", ",").split(","), dtype=float).reshape(-1, 3)[:, :2]
            dist = np.hypot((xy[:, 0] - bx) * length, (xy[:, 1] - by) * width)
            for side in (H, A):
                d = dist[team[:len(dist)] == side]
                d = d[~np.isnan(d)]
                if len(d):
                    near[frame, side] = d.min()

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, ball=ball, near=near)
    return ball, near


def tracking_log_evidence(ball, near, p=PARAMS):
    """(n, 3) log-likelihoods of the tracking under H, A, D. Zero rows mean 'no evidence'."""
    n = len(ball)
    out = np.zeros((n, 3))
    seen = ~np.isnan(ball[:, 0])
    m = p["outside_margin"]
    outside = seen & ((ball[:, 0] < -m) | (ball[:, 0] > 1 + m) |
                      (ball[:, 1] < -m) | (ball[:, 1] > 1 + m))

    dh = np.where(np.isnan(near[:, H]), 1e3, near[:, H])
    da = np.where(np.isnan(near[:, A]), 1e3, near[:, A])
    p_home = 1.0 / (1.0 + np.exp(-np.clip((da - dh) / p["margin_scale_m"], -30, 30)))
    closest = np.minimum(dh, da)
    t = np.clip((closest - p["near_m"]) / (p["far_m"] - p["near_m"]), 0.0, 1.0)
    p_alive = p["p_alive_near"] + t * (p["p_alive_far"] - p["p_alive_near"])
    p_alive = np.where(outside, p["p_alive_outside"], p_alive)
    p_home = np.where(outside, 0.5, p_home)

    eps = 1e-6
    out[seen, H] = np.log(p_alive[seen] * p_home[seen] + eps)
    out[seen, A] = np.log(p_alive[seen] * (1 - p_home[seen]) + eps)
    out[seen, D] = np.log(1 - p_alive[seen] + eps)
    return out * p["evidence_weight"]


# -- step 2: Smart Tagging priors -----------------------------------------------------------

def tagging_log_prior(events, n, home_name, fps, p=PARAMS):
    """(n, 3) log bonuses from the Smart Tagging intervals."""
    out = np.zeros((n, 3))
    pre, post = int(p["setpiece_pre_s"] * fps), int(p["setpiece_post_s"] * fps)
    for ev in events:
        code = ev["code"].strip().lower()
        start, end = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        if code == "set pieces":
            out[max(0, start - pre):min(n, start + post), D] += p["setpiece_dead_bonus"]
            continue
        team_name = (ev["tags"].get("Team") or "").strip().lower()
        if not team_name or team_name == "n/a":
            continue
        side = H if team_name == home_name.strip().lower() else A
        if code in DEFENDING_CODES:
            side = 1 - side
        elif code not in ATTACKING_CODES:
            continue
        out[start:end, side] += p["phase_team_bonus"]
    return out


# -- step 3: Viterbi decoding ---------------------------------------------------------------

def log_transitions(p=PARAMS):
    s, ad, da = p["p_switch_team"], p["p_alive_to_dead"], p["p_dead_to_alive"]
    t = np.array([[1 - s - ad, s, ad],
                  [s, 1 - s - ad, ad],
                  [da / 2, da / 2, 1 - da]])
    return np.log(t)


def viterbi(log_emit, log_trans):
    """Most likely state sequence for (n, k) log emissions and (k, k) log transitions."""
    n, k = log_emit.shape
    back = np.zeros((n, k), dtype=np.int8)
    score = log_emit[0] - np.log(k)
    for i in range(1, n):
        cand = score[:, None] + log_trans  # cand[from, to]
        back[i] = np.argmax(cand, axis=0)
        score = cand[back[i], np.arange(k)] + log_emit[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


# -- putting it together --------------------------------------------------------------------

def predict(game=GAME, data_dir=None, p=PARAMS, extra_log_emit=None):
    """Labels for every video frame: H/A/D inside the two halves, X outside them.

    extra_log_emit, an (n, 3) array, lets later versions add evidence before decoding.
    """
    meta_path = require(["atd_meta"], game, data_dir)[0]
    home_name = [t.findtext("Name") for t in ET.parse(meta_path).getroot().iter("Team")][0]
    fps = GAMES[game]["fps"]

    ball, near = read_tracking(game, data_dir)
    n = len(ball)
    log_emit = tracking_log_evidence(ball, near, p)
    log_emit += tagging_log_prior(load_smart_tagging.load(game, data_dir), n, home_name, fps, p)
    if extra_log_emit is not None:
        log_emit += extra_log_emit

    labels = np.full(n, "X", dtype="<U1")
    log_trans = log_transitions(p)
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = LABELS[viterbi(log_emit[first:last + 1], log_trans)]
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "possession_v1.csv"))
    args = parser.parse_args()

    labels = predict()
    path = write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("possession_v1  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, ROOT)))
