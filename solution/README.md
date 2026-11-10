# Our solution

## Reproduce the submission file

Python 3.10. With the data bundle unpacked in `./data` (or `MTAH_DATA_DIR` pointing at it):

```bash
pip install -r solution/requirements.txt
python solution/make_submission.py --team "<team name>"
```

This writes `evaluation/results/<team name>.csv`, runs the official format check, and prints the
file's SHA-256. Add `--fresh` to ignore `data/cache/` and re-parse the raw data (about a minute);
the output is byte-identical either way.

## What each file is

| file | what it adds |
|---|---|
| `possession_v1.py` | tracking evidence (who is nearest the ball), Smart Tagging priors, Viterbi decoding into whole possessions |
| `possession_v2.py` | dead-ball evidence: long ball gaps and slow player movement |
| `possession_v3.py` | close control counts more, flying balls count less, short ball gaps bridged |
| `possession_v4.py` | on unclear frames, lean towards the team defending that half |
| `possession_v5.py` | **final model**: a ball sitting still is evidence of a stoppage |
| `make_submission.py` | the one command above |
| `external_eval.py` | scores and tunes the models on external matches only |
| `idsse_convert.py` | converts an IDSSE (DFL) match into the compact form `external_eval.py` reads |

## How settings were chosen

Only on external data. The provided match's ground truth and video were never used for tuning.

* Metrica Sports' public sample games 1 and 2 (github.com/metrica-sports/sample-data), with ground
  truth derived from their event data by an algorithm. Downloaded automatically.
* IDSSE: three Bundesliga 2022/23 matches (Köln v Bayern, Bochum v Leverkusen, Düsseldorf v
  Nürnberg) from Bassek et al.,
  "An integrated dataset of spatiotemporal and event data in elite soccer", Sci Data 12, 195
  (2025), CC BY 4.0. Ground truth is the official DFL per-frame ball status and possession.
  Download the matchinformation and positions files from figshare, convert each with
  `python solution/idsse_convert.py <matchinformation.xml> <positions.xml> data/external/idsse/<id>.npz`.

`python solution/external_eval.py` prints the scores (spellF1):

| model | Metrica | IDSSE |
|---|---|---|
| v1 | 0.327 | 0.193 |
| v2 | 0.416 | 0.226 |
| v3 | 0.444 | 0.253 |
| v4 | 0.456 | 0.263 |
| v5 (before sharper team vote, 2 matches) | 0.485 | 0.337 |
| v5 (final, 3 matches) | 0.489 | 0.377 |
