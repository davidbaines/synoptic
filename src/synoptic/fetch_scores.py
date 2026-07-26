"""Recover a remote run's scores from its ClearML console log.

``generate`` echoes the full metrics table between ``METRICS_CSV_BEGIN`` and
``METRICS_CSV_END`` markers precisely so the console log — the one artefact
that always survives a remote run — carries the scores. This parses that
block back into a CSV.

    python -m synoptic.fetch_scores --task-id <id> --out experiments/scores-<run>.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd


def scores_from_log(log: str) -> pd.DataFrame:
    m = re.search(r"METRICS_CSV_BEGIN\n(.*?)METRICS_CSV_END", log, re.S)
    if not m:
        raise ValueError("no METRICS_CSV block in the console log")
    return pd.read_csv(StringIO(m.group(1)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover run scores from a console log")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--out", default=None,
                    help="output CSV (default experiments/scores-<task name>.csv)")
    args = ap.parse_args()

    from clearml import Task

    task = Task.get_task(task_id=args.task_id)
    # the server rejects windows above 10000 reports
    log = "".join(task.get_reported_console_output(number_of_reports=10000))
    try:
        table = scores_from_log(log)
    except ValueError as e:
        sys.exit(f"{e} (task {args.task_id}, status {task.status})")

    out = Path(args.out) if args.out else Path("experiments") / f"scores-{task.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    best = re.search(
        r"using best-validation checkpoint: chrF3_macro=([\d.]+) @ step (\d+)", log
    )
    print(f"WROTE {out} ({len(table)} rows, task status {task.status})")
    if best:
        print(f"best validation: chrF3_macro={best.group(1)} @ step {best.group(2)}")


if __name__ == "__main__":
    main()
