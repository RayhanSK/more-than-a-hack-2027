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
| `possession_v4.py` | **final model**: on unclear frames, lean towards the team defending that half |
| `make_submission.py` | the one command above |
| `external_eval.py` | scores and tunes the models on external matches only |

## How settings were chosen

Only on external data: Metrica Sports' public sample games 1 and 2
(github.com/metrica-sports/sample-data), with ground truth derived from their event data by an
algorithm. `python solution/external_eval.py` downloads them and prints the scores. The provided
match's ground truth and video were never used for tuning.

| model | spellF1 (external) |
|---|---|
| v1 | 0.327 |
| v2 | 0.416 |
| v3 | 0.444 |
| v4 | 0.456 |
