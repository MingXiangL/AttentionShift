# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
from tempfile import TemporaryDirectory
import time 

import torch
from torch.optim import Optimizer

import mmcv
from mmcv.runner import RUNNERS, EpochBasedRunner
from .checkpoint import save_checkpoint
import mmcv
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import weights_to_cpu, get_state_dict, load_state_dict

try:
    import apex
except:
    print('apex is not installed')


def save_checkpoint_free(model, filename, optimizer=None, meta=None, **kwargs):
    """Save checkpoint to file.

    The checkpoint will have 4 fields: ``meta``, ``state_dict`` and
    ``optimizer``, ``amp``. By default ``meta`` will contain version
    and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()
    for k in kwargs:
        checkpoint[k] = kwargs[k]
    # save amp state dict in the checkpoint
    checkpoint['amp'] = apex.amp.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()



@RUNNERS.register_module()
class EpochBasedRunnerAmpTS(EpochBasedRunner):
    """Epoch-based Runner with AMP support.

    This runner train models epoch by epoch.
    """

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        teacher_state_dict = weights_to_cpu(get_state_dict(self.model.module.tw.teacher))
        save_checkpoint_free(self.model, filepath, optimizer=optimizer, meta=meta, teacher_state_dict=teacher_state_dict)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
                teacher = getattr(self.model.module, 'tw', None)
            else:
                checkpoint = self.load_checkpoint(checkpoint)
                teacher = getattr(self.model, 'tw', None)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
            teacher = getattr(self.model, 'tw', None)
        
        if 'teacher_state_dict' in checkpoint:
            load_state_dict(teacher.teacher, checkpoint['teacher_state_dict'])

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint:
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')
        
        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
