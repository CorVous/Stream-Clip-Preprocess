"""CLI subcommands for stream-clip-preprocess.

Each subcommand is defined in its own module and exports:
- add_parser(subparsers, common_parser): Register the subcommand
- run(args): Execute the subcommand logic
"""

from stream_clip_preprocess.commands import version as version_cmd

__all__ = ["version_cmd"]
