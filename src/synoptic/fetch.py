"""Download a remote ClearML run's artifacts to a local checkpoint directory.

    uv run python -m synoptic.fetch --name ie_base_shareable

When training runs on a remote agent (queue `jobs_backlog`), the checkpoint and
generated artefacts live on that machine and are uploaded to ClearML as the
`run` artifact. This pulls them back into `checkpoints/<name>/` so the local
publish pipeline (`synoptic.publish`) can use them exactly as if the run had
happened here.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .data import repo_root


def fetch(task_id: str | None, name: str | None, output: Path) -> Path:
    from clearml import Task

    if task_id:
        task = Task.get_task(task_id=task_id)
    else:
        task = Task.get_task(project_name="bible-interlingua", task_name=name)
    if task is None:
        raise SystemExit(f"No ClearML task found for {task_id or name!r}")
    print(f"Task {task.id} '{task.name}' status={task.get_status()}")

    if "run" not in task.artifacts:
        raise SystemExit(
            f"Task has no 'run' artifact (artifacts: {list(task.artifacts)}). "
            "Was it trained with --generate-after / artifact upload enabled?"
        )
    local = Path(task.artifacts["run"].get_local_copy())
    output.mkdir(parents=True, exist_ok=True)
    # get_local_copy returns the extracted folder; copy its contents in
    for item in local.iterdir():
        dest = output / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    print(f"Fetched artifacts into {output}")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch a ClearML run's artifacts locally")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-id")
    g.add_argument("--name", help="task name (most recent match in project synoptic)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else repo_root() / "checkpoints" / (args.name or args.task_id))
    fetch(args.task_id, args.name, out)


if __name__ == "__main__":
    main()
