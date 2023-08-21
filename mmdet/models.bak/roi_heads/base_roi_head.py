from abc import ABCMeta, abstractmethod

import torch.nn as nn

from ..builder import build_shared_head, build_head


class BaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 mil_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 mae_head=None,
                 bbox_rec_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
            
        if mil_head is not None:
            self.init_mil_head(bbox_roi_extractor, mil_head)
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)
        if bbox_rec_head is not None:
            self.init_bbox_rec_head(bbox_roi_extractor, bbox_rec_head)
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()
        
        if mae_head is not None:
            self.mae_head = build_head(mae_head)
            self.with_mae_head = True
        else:
            self.with_mae_head = False
    @property
    def with_mil(self):
        """bool: whether the RoI head contains a `mil_head`"""
        return hasattr(self, 'mil_head') and self.mil_head is not None
    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
    @property
    def with_bbox_rec(self):
        """bool: whether the RoI head contains a `bbox_rec_head`"""
        return hasattr(self, 'bbox_rec_head') and self.bbox_rec_head is not None
    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass
    @abstractmethod
    def init_bbox_rec_head(self):
        """Initialize ``bbox_rec_head``"""
        pass
    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self, x, img_meta, **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
