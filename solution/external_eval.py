"""Score and tune the possession models on EXTERNAL matches -- never on the competition match.

    python solution/external_eval.py              # download if needed, score v1 and v2
    python solution/external_eval.py --tune       # small grid search, prints the best settings

The competition rules say every tunable setting must be chosen on data from other matches. This
script uses Metrica Sports' public sample data (github.com/metrica-sports/sample-data, two full
matches of tracking and event data; credit: Metrica Sports). Nothing here reads the competition
data, its ground truth or its video.

Ground truth for these matches is derived by an algorithm from their event data, not by hand:

  * a SET PIECE by team T restarts play: T has the ball from its start frame;
  * BALL OUT, FAULT RECEIVED, and a SHOT that goes out or in, stop play: dead (D) from there until
    the next team event;
  * any other team event (PASS, RECOVERY, BALL LOST, SHOT) means that team has the ball from its
    start frame;
  * CHALLENGE and CARD are skipped: a contested ball stays with the last team that had it, which
    is the competition's own rule.

The sample tracking differs from the competition's automatic tracking in two ways, so every model
is scored under two scenarios and the tuning target is their mean:

  * "raw": the sample's own ball gaps are kept. Metrica leaves the ball out almost every time play
    stops, which makes "a long gap means a dead ball" look better than it can be on our match.
  * "filled": the sample's gaps are interpolated away first, so a gap says nothing about a dead
    ball. This is the pessimistic case for the gap signal.

In both, the ball is then removed in random bursts until about 40% of frames have no ball, the
share the competition documentation reports. The burst pattern is seeded, so every run is
identical.
"""

import argparse
import itertools
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import possession_v1 as v1  # noqa: E402
import possession_v2 as v2  # noqa: E402
import possession_v3 as v3  # noqa: E402

H, A, D = v1.H, v1.A, v1.D
EXTERNAL_DIR = os.path.join(ROOT, "data", "external", "metrica_sample")  # gitignored
SOURCE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2)
PITCH = (105.0, 68.0)  # the sample data's documented pitch size
FPS = 25

# Degradation: target share of frames with no ball, and mean burst length in frames.
MISSING_SHARE = 0.4
MISSING_BURST = 40
SEEDS = (0, 1)
SCENARIOS = ("raw", "filled")


# -- data -----------------------------------------------------------------------------------

def download(game):
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    paths = {}
    for part in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team"):
        name = "Sample_Game_{}_{}.csv".format(game, part)
        path = os.path.join(EXTERNAL_DIR, name)
        if not os.path.exists(path):
            print("downloading {}".format(name))
            urllib.request.urlretrieve("{}/Sample_Game_{}/{}".format(SOURCE, game, name), path)
        paths[part] = path
    return paths


def read_tracking_csv(path):
    """(frames, period, players (n, k, 2), ball (n, 2)) from a Metrica CSV tracking file."""
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    xy_cols = [c for c in cols[3:]]
    values = df[xy_cols].to_numpy(dtype=float)
    ball_at = cols.index("Ball") - 3
    ball = values[:, ball_at:ball_at + 2]
    players = np.delete(values, [ball_at, ball_at + 1], axis=1).reshape(len(df), -1, 2)
    return df["Frame"].to_numpy(), df["Period"].to_numpy(), players, ball


def event_labels(events, n_frames):
    """Per-frame H/A/D codes derived from the event data, as described in the module docstring.
    Frames before the first event are D. Index is frame - 1."""
    changes = []
    for row in events.sort_values("Start Frame", kind="stable").itertuples(index=False):
        kind = str(row.Type).upper()
        sub = str(row.Subtype).upper()
        team = H if str(row.Team).lower() == "home" else A
        start, end = int(row._4), int(row._6)
        if kind in ("CHALLENGE", "CARD"):
            continue
        if kind in ("BALL OUT", "FAULT RECEIVED"):
            changes.append((start, D))
        elif kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
            changes.append((start, team))
            changes.append((max(end, start), D))
        else:
            changes.append((start, team))
    codes = np.full(n_frames, D, dtype=np.int8)
    for (frame, code), nxt in zip(changes, changes[1:] + [(n_frames + 1, None)]):
        codes[max(0, frame - 1):max(0, nxt[0] - 1)] = code
    return codes


def fill_gaps(ball, period):
    """Linear interpolation of missing ball positions within each period (ends held)."""
    out = ball.copy()
    for half in np.unique(period):
        idx = np.flatnonzero(period == half)
        for k in range(2):
            col = out[idx, k]
            ok = ~np.isnan(col)
            if ok.any():
                out[idx, k] = np.interp(np.arange(len(idx)), np.flatnonzero(ok), col[ok])
    return out


def degrade(ball, seed):
    """Remove the ball in random bursts until about MISSING_SHARE of frames have none."""
    rng = np.random.default_rng(seed)
    out = ball.copy()
    n = len(out)
    already = np.isnan(out[:, 0]).mean()
    need = max(0.0, MISSING_SHARE - already)
    n_bursts = int(need * n / MISSING_BURST)
    for start in rng.integers(0, n, n_bursts):
        length = int(rng.exponential(MISSING_BURST)) + 1
        out[start:start + length] = np.nan
    return out


def nearest(players_h, players_a, ball):
    """(n, 2) distance in metres from the ball to the nearest home and away player."""
    length, width = PITCH
    near = np.full((len(ball), 2), np.nan)
    for side, players in ((H, players_h), (A, players_a)):
        dist = np.hypot((players[:, :, 0] - ball[:, None, 0]) * length,
                        (players[:, :, 1] - ball[:, None, 1]) * width)
        with np.errstate(all="ignore"):
            near[:, side] = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
        near[np.isinf(near[:, side]), side] = np.nan
    return near


def player_speed(players_h, players_a):
    """(n,) median player speed in m/s, in the same form as the competition pipeline."""
    length, width = PITCH
    allp = np.concatenate([players_h, players_a], axis=1)
    step = np.hypot(np.diff(allp[:, :, 0], axis=0) * length,
                    np.diff(allp[:, :, 1], axis=0) * width) * FPS
    with np.errstate(all="ignore"):
        return np.concatenate([[np.nan], np.nanmedian(step, axis=1)])


def load_game(game):
    paths = download(game)
    frames, period, ph, ball = read_tracking_csv(paths["RawTrackingData_Home_Team"])
    _, _, pa, _ = read_tracking_csv(paths["RawTrackingData_Away_Team"])
    gt = event_labels(pd.read_csv(paths["RawEventsData"]), len(frames))
    return {"period": period, "home": ph, "away": pa, "ball": ball, "gt": gt,
            "speed": player_speed(ph, pa)}


# -- scoring --------------------------------------------------------------------------------

def decode(g, ball, p, version):
    period, speed = g["period"], g["speed"]
    if version >= 3:
        blocks = [(i[0], i[-1]) for i in (np.flatnonzero(period == h) for h in np.unique(period))]
        filled = v3.fill_short_gaps(ball, int(round(p["fill_max_s"] * FPS)), blocks)
        near = nearest(g["home"], g["away"], filled)
        log_emit = v3.log_evidence(ball, filled, near, speed, PITCH, FPS, p)
    else:
        log_emit = v1.tracking_log_evidence(ball, nearest(g["home"], g["away"], ball), p)
    if version == 2:
        log_emit += v2.dead_log_evidence(ball, speed, FPS, p)
    out = np.empty(len(ball), dtype=np.int8)
    log_trans = v1.log_transitions(p)
    for half in np.unique(period):
        idx = np.flatnonzero(period == half)
        out[idx] = v1.viterbi(log_emit[idx], log_trans)
    return out


def score(games, p, version, scenarios=SCENARIOS):
    """Mean spellF1 and macroF1 over every game, scenario and degradation seed."""
    spell, macro = [], []
    for g, scenario, seed in itertools.product(games, scenarios, SEEDS):
        base = g["ball"] if scenario == "raw" else fill_gaps(g["ball"], g["period"])
        ball = degrade(base, seed)
        pred = decode(g, ball, p, version)
        # Separate the halves with one ungraded frame so no possession spans halftime.
        cut = np.flatnonzero(np.diff(g["period"]))[0] + 1
        gt_l = np.insert(v1.LABELS[g["gt"]], cut, "X")
        pr_l = np.insert(v1.LABELS[pred], cut, "X")
        gt_c, pr_c = pm.label_codes(gt_l), pm.label_codes(pr_l)
        runs = pm.valid_runs(gt_c)
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(gt_c, pr_c))))
        spell.append(pm.spell_scores(gt_c, pm.fill_abstentions(pr_c, runs), runs)["mean_f1"])
    return float(np.mean(spell)), float(np.mean(macro))


GRID = {
    "evidence_weight": [0.05, 0.1, 0.2],
    "p_switch_team": [1 / 125, 1 / 250, 1 / 500],
    "gap_mid_s": [2.0, 4.0, 8.0],
    "slow_mps": [1.0, 1.5, 2.0],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", action="store_true", help="grid-search v2's main settings")
    args = parser.parse_args()

    games = [load_game(g) for g in GAMES]
    for g, data in zip(GAMES, games):
        counts = np.bincount(data["gt"], minlength=3)
        print("game {}: {} frames  H {}  A {}  D {}".format(g, len(data["gt"]), *counts))

    print("\n{:<8} {:<8} {:>8} {:>8}".format("model", "scenario", "spellF1", "macroF1"))
    for name, params, version in (("v1", v1.PARAMS, 1), ("v2", v2.PARAMS, 2),
                                  ("v3", v3.PARAMS, 3)):
        for scenario in SCENARIOS + ("mean",):
            which = SCENARIOS if scenario == "mean" else (scenario,)
            print("{:<8} {:<8} {:>8.4f} {:>8.4f}".format(
                name, scenario, *score(games, params, version, which)))

    if args.tune:
        keys = list(GRID)
        results = []
        for combo in itertools.product(*(GRID[k] for k in keys)):
            p = dict(v2.PARAMS)
            p.update(zip(keys, combo))
            p["dead_evidence_weight"] = p["evidence_weight"]
            results.append((score(games, p, 2), dict(zip(keys, combo))))
        results.sort(key=lambda r: -r[0][0])
        print("\nbest settings on external data (spellF1, macroF1):")
        for (s, m), combo in results[:5]:
            p = dict(v2.PARAMS)
            p.update(combo)
            p["dead_evidence_weight"] = p["evidence_weight"]
            split = "  ".join("{} {:.4f}".format(sc, score(games, p, 2, (sc,))[0])
                              for sc in SCENARIOS)
            print("  {:.4f} {:.4f}  {}  ({})".format(s, m, combo, split))
