"""Convert one IDSSE (DFL) positions file into a compact .npz for external_eval.py.

    python solution/idsse_convert.py <matchinformation.xml> <positions_raw_observed.xml> <out.npz>

IDSSE: Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer." Sci Data 12, 195 (2025), CC BY 4.0. Seven Bundesliga matches with
official DFL tracking at 25 fps. Every ball frame carries the official BallStatus (1 alive, 0
dead) and BallPossession (1 home, 2 guest), which become the ground truth here: H/A from
BallPossession while alive, D while dead.

The files are large XML; this reads them line by line with a regex, so memory stays small.
Coordinates are converted to the competition's form: normalised 0..1 along the pitch.
"""

import re
import sys
from array import array

import numpy as np

ATTR = re.compile(r'(\w+)="([^"]*)"')


def main(meta_path, pos_path, out_path):
    meta = open(meta_path, encoding="utf-8").read()
    teams = {m.group(1): m.group(2) for m in
             re.finditer(r'<Team TeamId="([^"]+)"[^>]*Role="(home|guest)"', meta)}
    length = float(re.search(r'PitchX="([\d.]+)"', meta).group(1))
    width = float(re.search(r'PitchY="([\d.]+)"', meta).group(1))

    # section -> {"ball": [N, x, y, status, poss], "home": {pid: [N, x, y]}, "guest": ...},
    # each list a compact array.array so a 400 MB file fits in little memory.
    data = {}
    current = None
    frame = re.compile(r'N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
    ball_flags = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame ") and current is not None:
                m = frame.search(line)
                current[0].append(int(m.group(1)))
                current[1].append(float(m.group(2)))
                current[2].append(float(m.group(3)))
                if len(current) == 5:
                    f = ball_flags.search(line)
                    current[3].append(int(f.group(2)) if f else 0)
                    current[4].append(int(f.group(1)) if f else 0)
            elif line.startswith("<FrameSet"):
                a = dict(ATTR.findall(line))
                sec = data.setdefault(a["GameSection"], {"ball": None, "home": {}, "guest": {}})
                team = a["TeamId"]
                if team == "BALL":
                    current = sec["ball"] = [array("i"), array("f"), array("f"), array("b"),
                                             array("b")]
                elif team in teams:
                    current = sec[teams[team]].setdefault(
                        a["PersonId"], [array("i"), array("f"), array("f")])
                else:
                    current = None

    out = {"pitch": np.array([length, width])}
    for k, name in enumerate(("firstHalf", "secondHalf")):
        sec = data[name]
        n_b, bx, by, status, poss = (np.asarray(v) for v in sec["ball"])
        order = np.argsort(n_b)
        frames = n_b[order]
        out["ball{}".format(k)] = np.column_stack([(bx[order] + length / 2) / length,
                                                   (by[order] + width / 2) / width])
        out["status{}".format(k)] = status[order].astype(np.int8)
        out["poss{}".format(k)] = poss[order].astype(np.int8)
        for side in ("home", "guest"):
            players = sec[side]
            arr = np.full((len(frames), len(players), 2), np.nan, dtype=np.float32)
            for j, (n, x, y) in enumerate(players.values()):
                i = np.searchsorted(frames, np.asarray(n))
                ok = (i < len(frames)) & (frames[np.minimum(i, len(frames) - 1)] == np.asarray(n))
                arr[i[ok], j, 0] = (np.asarray(x)[ok] + length / 2) / length
                arr[i[ok], j, 1] = (np.asarray(y)[ok] + width / 2) / width
            out["{}{}".format(side, k)] = arr
    np.savez_compressed(out_path, **out)
    print("wrote", out_path, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
