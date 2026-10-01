from __future__ import annotations

import argparse
import sys

from .list import add_list_parser
from .tps import add_tps_parser
from .transformers_compat import dispatch_transformers_cli, is_transformers_cli_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transformers-mblt",
        description=(
            "Mobilint NPU helpers for Hugging Face Transformers. Upstream Transformers commands such as "
            "`chat`, `serve`, `download`, `env`, and `version` are delegated to the installed `transformers` "
            "package with Mobilint models registered."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    commands_parser = parser.add_subparsers(help="transformers-mblt command helpers")

    add_list_parser(commands_parser)
    add_tps_parser(commands_parser)

    return parser


def _package_version() -> str:
    from .. import __version__

    return __version__


def main() -> int:
    # Upstream Transformers commands own their argument parsing, so delegate before argparse sees them.
    if is_transformers_cli_command(sys.argv):
        return dispatch_transformers_cli(sys.argv)

    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "_handler"):
        return args._handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
