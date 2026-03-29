#!/usr/bin/env python3
"""
Run finetune_rbp.py for one variant across ALL datasets roots.

This wrapper discovers task folders that contain train/dev/test CSV and invokes
finetune_rbp.py with explicit --data_root, so datasets from different roots
can be evaluated together safely.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ALL_VARIANTS = [
      "dnabert2_tapt_v4_5132",
      "dnabert2_tapt_v4_28226"
]


@dataclass
class Task:
    source: str
    rbp_name: str
    data_root: Path


REQUIRED_SPLITS = ("train.csv", "dev.csv", "test.csv")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fine-tune one variant on all dataset roots")
    parser.add_argument("--variant", required=True, choices=ALL_VARIANTS)
    parser.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            str(base / "data" / "finetune_data_koo"),
            str(base / "DNABERT2" / "data"),
            str(base / "data" / "diff_cells_data" / "splits_csv"),
        ],
        help="Roots containing per-task folders with train/dev/test CSV",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(base / "finetune_rbp" / "results" / "rbp_all_datasets"),
        help="Base output folder; source and variant subdirs are created below this path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=0,
        help="If >0, run only first N discovered tasks (debug/smoke usage)",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining tasks even if one task fails",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Run only tasks missing results.json under output_root/source/variant/rbp_name.",
    )
    return parser.parse_args()


def discover_tasks(data_roots: List[str]) -> List[Task]:
    tasks: List[Task] = []
    seen = set()

    for root_str in data_roots:
        root = Path(root_str)
        if not root.exists() or not root.is_dir():
            print(f"[WARN] missing root: {root}")
            continue
        source = root.name
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if not all((d / fname).exists() for fname in REQUIRED_SPLITS):
                continue
            key = (source, d.name)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(Task(source=source, rbp_name=d.name, data_root=root))

    return tasks


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    finetune_script = script_dir / "finetune_rbp.py"

    tasks = discover_tasks(args.data_roots)
    if not tasks:
        raise RuntimeError("No valid tasks discovered across data_roots")

    if args.only_missing:
        filtered: List[Task] = []
        skipped = 0
        for task in tasks:
            result_json = (
                Path(args.output_root)
                / task.source
                / args.variant
                / task.rbp_name
                / "results.json"
            )
            if result_json.exists():
                skipped += 1
                continue
            filtered.append(task)
        tasks = filtered
        print(f"[run] only_missing enabled: skipped_completed={skipped}")

    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    print(f"[run] variant={args.variant}")
    print(f"[run] discovered tasks={len(tasks)}")

    failures = 0
    for idx, task in enumerate(tasks, start=1):
        output_dir = Path(args.output_root) / task.source
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(finetune_script),
            "--variant",
            args.variant,
            "--rbp_name",
            task.rbp_name,
            "--data_root",
            str(task.data_root),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ]

        print("-" * 80)
        print(f"[{idx}/{len(tasks)}] {task.source}/{task.rbp_name}")
        print(" ".join(cmd))

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures += 1
            print(f"[ERROR] failed task: {task.source}/{task.rbp_name} (exit={proc.returncode})")
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)

    if failures > 0:
        raise SystemExit(f"Completed with failures: {failures}")

    print("[done] all tasks finished successfully")


if __name__ == "__main__":
    main()
