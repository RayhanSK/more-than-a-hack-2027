"""Regenerate the submission file from the provided data with one command.

    python solution/make_submission.py --team "<team name>"

Runs the final model (possession_v9, with solution/v9_weights.json) from the raw ATD tracking
and Smart Tagging, writes
evaluation/results/<team name>.csv, runs the official format check on it, and prints the file's
SHA-256 so two runs can be compared. Point MTAH_DATA_DIR at the data bundle if it is not in ./data.

--fresh ignores data/cache/ and re-parses everything into a temporary folder (slower, same
result).
"""

import argparse
import hashlib
import os
import tempfile
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_v1 as v1  # noqa: E402
import possession_v9 as final  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is named after it")
    parser.add_argument("--fresh", action="store_true", help="ignore data/cache/, re-parse all")
    args = parser.parse_args()

    if args.fresh:
        v1.CACHE_DIR = tempfile.mkdtemp(prefix="mtah_cache_")

    labels = final.predict()
    path = v1.write_submission(os.path.join(v1.RESULTS_DIR, "{}.csv".format(args.team)), labels)
    counts = {label: int((labels == label).sum()) for label in "HADX"}
    print("labels  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(path, v1.ROOT)))

    check = subprocess.run([sys.executable, os.path.join(v1.ROOT, "evaluation",
                                                         "validate_submission.py"), path])
    with open(path, "rb") as handle:
        print("sha256 {}".format(hashlib.sha256(handle.read()).hexdigest()))
    return check.returncode


if __name__ == "__main__":
    sys.exit(main())
