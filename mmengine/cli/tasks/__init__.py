# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction


class Task:
    """Abstract base class for all command jobs."""

    command: str = 'unknown'
    description: str | None = None

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: ArgumentParser):
        """Add arguments to this command."""

    @abstractmethod
    def run(self, args: Namespace) -> int:
        """Run this command."""
        return 0

    def __init__(self, subparsers: _SubParsersAction):
        parser = subparsers.add_parser(self.command, description=self.description)
        self.add_arguments(parser)
