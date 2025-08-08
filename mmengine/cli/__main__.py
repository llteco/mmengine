# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from ..version import __version__
from .command import DatasetCommand


def main():
    """Entry of mmengine command line interface (CLI)"""
    parser = ArgumentParser(
        'mmengine-cli',
        usage='mmengine-cli <command> [<args>]',
        description='MMEngine Command Line Interface (CLI)',
    )
    parser.add_argument(
        '--version', '-V', action='version', version=f"%(prog)s {__version__}"
    )
    subparser = parser.add_subparsers(help='mmengine-cli command helpers')

    DatasetCommand.register_subcommand(subparser)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    args.func.run(args)


if __name__ == '__main__':
    main()
