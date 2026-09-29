"""``transformers-mblt list`` command: list Mobilint Transformers models and tasks."""

from __future__ import annotations

import argparse
import json
from typing import Any


def add_list_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``list`` subcommand."""
    from ..utils.api import list_tasks

    parser = subparsers.add_parser(
        "list",
        help="List Mobilint models on the Hugging Face Hub (falls back to the local cache)",
        description=(
            "List Mobilint Transformers models grouped by pipeline task. Queries the Hugging Face Hub and falls "
            "back to the local Hugging Face cache when the Hub is unreachable."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=list_tasks(),
        default=None,
        help="Pipeline task to list. Repeat to list several tasks. Default: every supported task.",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private repositories visible to the current Hugging Face token.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--tasks", action="store_true", help="Only print the supported task names.")
    parser.set_defaults(_handler=_cmd_list)
    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    from ..utils.api import list_models, list_tasks

    if args.tasks:
        tasks = list_tasks()
        print(json.dumps(tasks, indent=2) if args.json else "\n".join(tasks))
        return 0

    models = list_models(args.task or list_tasks(), include_private=args.include_private)
    if args.json:
        print(json.dumps(models, indent=2))
        return 0
    for task, model_ids in models.items():
        print(f"{task} ({len(model_ids)})")
        for model_id in model_ids:
            print(f"  {model_id}")
    return 0
