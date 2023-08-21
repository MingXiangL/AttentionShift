import mmcv
from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class LossWeightAdjustHook(Hook):
    def __init__(self, start_epoch=1, **kwargs):
        self.start_epoch = start_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        loss_weight_mask = epoch > self.start_epoch
        loss_weight_bbox = epoch > self.start_epoch
        runner.model.module.roi_head.mask_head.loss_weight_mask_start = loss_weight_mask
        runner.model.module.roi_head.bbox_head.loss_weight_bbox_start = loss_weight_bbox
