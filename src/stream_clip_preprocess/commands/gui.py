"""GUI subcommand - launch the main application window."""

import argparse
import logging

_logger = logging.getLogger(__name__)


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the gui subcommand.

    :param subparsers: Subparser action from main parser
    :param common_parser: Parent parser with common arguments
    :return: The created subparser
    """
    parser = subparsers.add_parser(
        "gui",
        parents=[common_parser],
        help="Launch the GUI",
        description="Launch the Stream Clip Preprocess GUI",
    )
    parser.set_defaults(func=run)
    return parser


def run(_args: argparse.Namespace) -> int:
    """Execute the gui command.

    :param _args: Parsed command-line arguments (unused)
    :return: Exit code (0 for success)
    """
    _logger.info("Launching GUI")
    from stream_clip_preprocess.gui.app import launch  # noqa: PLC0415

    launch()
    return 0
