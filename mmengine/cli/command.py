# Copyright (c) OpenMMLab. All rights reserved.
from . import BaseCLICommand
from .tasks.browse_dataset import BrowseDatasetTask


class DatasetCommand(BaseCLICommand):
    """Dataset Interface (CLI)"""

    command = 'dataset'
    jobs = [BrowseDatasetTask]
