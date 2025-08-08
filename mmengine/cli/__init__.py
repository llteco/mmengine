# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC
from argparse import Namespace, _SubParsersAction

from .tasks import Task

_JOBS: dict[str, Task] = {}


class BaseCLICommand(ABC):
    """Abstract CLI command class."""

    command: str = 'undefined'
    jobs: list[type[Task]] = []

    @classmethod
    def register_subcommand(cls, parser: _SubParsersAction):
        r"""An abstract static method to add a command to subparsers.

        Example:

            class DatasetCommand(BaseCLICommand):
                command = "dataset"
                jobs = [TaskCls1, TaskCls2]
        """
        subparser = parser.add_parser(
            cls.command,
            usage=f"{cls.command} <command> [<args>]",
            description=f"{cls.command.upper()} Command Line Interface (CLI)",
        )
        gpu_parser = subparser.add_subparsers(
            help=f"{cls.command} helpers", dest='command'
        )
        subparser.set_defaults(func=cls)
        for job in cls.jobs:
            cls.add_job(job(gpu_parser))

    @staticmethod
    def add_job(job: Task):
        """Add a specific job to the command."""
        _JOBS[job.command] = job

    @staticmethod
    def run(args: Namespace):
        """Run the command."""

        if args.command not in _JOBS:
            print(
                'Error: missing a valid command. '
                f"Choose from: {sorted(_JOBS.keys())}"
            )
            exit(1)

        ret = _JOBS[args.command].run(args)
        raise SystemExit(ret)
