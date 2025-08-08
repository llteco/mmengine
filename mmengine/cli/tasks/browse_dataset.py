# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
from argparse import ArgumentParser, Namespace

from mmengine import ProgressBar, init_default_scope
from mmengine.config import Config, DictAction
from mmengine.registry import DATASETS, VISUALIZERS
from . import Task


class BrowseDatasetTask(Task):
    """Browse the dataset defined in a config file."""

    command = 'browse'

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        parser.add_argument('config', help='train config file path')
        parser.add_argument(
            '--save-dir',
            default=None,
            type=str,
            help='If there is no display interface, you can save it',
        )
        parser.add_argument(
            '--show', action='store_true', help='display the visualization results'
        )
        parser.add_argument(
            '--show-interval', type=float, default=2, help='the interval of show (s)'
        )
        parser.add_argument(
            '--max-count', type=int, default=100, help='max count of samples to browse'
        )
        parser.add_argument(
            '--mode',
            default='train',
            choices=['train', 'val', 'test'],
            help='choose a type of dataset to browse. Defaults to train',
        )
        parser.add_argument(
            '--cfg-options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.',
        )

    def _visual_mmdet(self, args, visualizer, idx, item):
        data_sample = item['data_samples']
        inputs = item['inputs']
        for img_idx, img_data_sample in enumerate(data_sample):
            img_path = img_data_sample.img_path
            img = inputs[img_idx].permute(1, 2, 0).numpy()
            out_file = (
                osp.join(args.save_dir, str(idx).zfill(6), f"img_{img_idx}.jpg")
                if args.save_dir is not None
                else None
            )
            img = img[..., [2, 1, 0]]  # bgr to rgb
            visualizer.add_datasample(
                osp.basename(img_path),
                img,
                data_sample=img_data_sample,
                draw_pred=False,
                show=args.show,
                wait_time=args.show_interval,
                out_file=out_file,
            )
            # Record file path mapping.
            if args.save_dir is not None:
                with open(
                    osp.join(args.save_dir, str(idx).zfill(6), 'info.txt'),
                    'a',
                    encoding='utf-8',
                ) as f:
                    f.write(
                        f"The source filepath of img_{img_idx}.jpg"
                        f"is `{img_path}`.\n"
                    )

    def _visual_mmseg(self, args, visualizer, idx, item):
        inputs = item['inputs']
        if inputs.ndim == 3:
            imgs = item['inputs'].permute(1, 2, 0).numpy()
            imgs = imgs[None]  # expand to [1, H, W, C]
        elif inputs.ndim == 4:
            imgs = item['inputs'].permute(0, 2, 3, 1).numpy()
        else:
            raise ValueError(f"inputs ndim should be 3 or 4, but got {inputs.ndim}")
        imgs = imgs[..., [2, 1, 0]]  # bgr to rgb
        data_sample = item['data_samples'].numpy()
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = (
            osp.join(args.save_dir, osp.basename(img_path))
            if args.save_dir is not None
            else None
        )

        has_gt = 'gt_sem_seg' in data_sample
        if has_gt:
            gt_sem_seg = data_sample.gt_sem_seg.data.copy()
        for idx, img in enumerate(imgs):
            if has_gt:
                data_sample.gt_sem_seg.data = gt_sem_seg[:, idx]  # type: ignore
            visualizer.add_datasample(
                name=osp.basename(img_path) + f"%{idx}",
                image=img.copy(),
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=False,
                wait_time=args.show_interval,
                out_file=out_file,
                show=args.show,
            )

    def run(self, args: Namespace) -> int:
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        scope = cfg.get('default_scope', 'mmdet')
        init_default_scope(scope)
        importlib.import_module('.datasets', scope)

        if scope in DATASETS.children:
            dataset = DATASETS.children[scope].build(
                cfg[f"{args.mode}_dataloader"].dataset
            )
        else:
            dataset = DATASETS.build(cfg[f"{args.mode}_dataloader"].dataset)

        if scope in VISUALIZERS.children:
            visualizer = VISUALIZERS.children[scope].build(cfg.visualizer)
        else:
            visualizer = VISUALIZERS.build(cfg.visualizer)
        visualizer.dataset_meta = dataset.metainfo

        progress_bar = ProgressBar(len(dataset))
        for idx, item in enumerate(dataset):  # inputs data_samples
            if scope == 'mmdet':
                self._visual_mmdet(args, visualizer, idx, item)
            elif scope == 'mmseg':
                self._visual_mmseg(args, visualizer, idx, item)
            else:
                raise NotImplementedError(
                    f"Browsing for {scope} is not implemented. "
                    f"You could edit this file {__file__} by adding a function "
                    'like `_visual_mmseg`'
                )
            progress_bar.update()
            if idx >= args.max_count - 1:
                break
        return 0
