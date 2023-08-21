from ..builder import DETECTORS
from .two_stage_point_align import TwoStageDetectorPointSupAlign

@DETECTORS.register_module()
class FasterRCNNPointSupAlign(TwoStageDetectorPointSupAlign):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 *args, **kwargs):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            *args, **kwargs)
