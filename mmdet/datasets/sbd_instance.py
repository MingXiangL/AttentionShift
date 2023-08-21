from collections import OrderedDict
from tkinter.messagebox import NO
import os.path as osp
from pycocotools import mask
import pdb
import scipy
import os

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.xml_style import XMLDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label
import numpy as np

@DATASETS.register_module()
class SBDDatasetInstance(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.year = 2012

    # copied from chainerCV at  https://github.com/chainer/chainercv/blob/master/chainercv/datasets/sbd/sbd_instance_segmentation_dataset.py
    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.img_prefix, 'SBDSegmentationObject', 'cls', data_id + '.mat')
        inst_file = os.path.join(
            self.img_prefix, 'SBDSegmentationObject', 'inst', data_id + '.mat')
        label_anno = scipy.io.loadmat(label_file)
        label_img = label_anno['GTcls']['Segmentation'][0][0].astype(np.int32)
        inst_anno = scipy.io.loadmat(inst_file)
        inst_img = inst_anno['GTinst']['Segmentation'][0][0].astype(np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img

    
    def get_ann_info(self, idx):
        # ann0 = super().get_ann_info(idx)
        img_id = self.data_infos[idx]['id']
        label_img, inst_img = self._load_label_inst(img_id)
        mask_bin, labels = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        mask_bin = np.asfortranarray(mask_bin.transpose(1,2,0).astype(np.uint8))
        mask_encode = mask.encode(mask_bin)
        bbox_mask   = mask.toBbox(mask_encode)
        bbox_mask[:, 2:] += bbox_mask[:, :2]
        points = (bbox_mask[:, :2] + bbox_mask[:, 2:]) / 2
        ann = dict(
            bboxes=bbox_mask.astype(np.float32),
            labels=labels.astype(np.int64),
            points=points.astype(np.float32),
            bboxes_ignore=np.zeros((0, 4)).astype(np.float32),
            labels_ignore=np.zeros((0, )).astype(np.int64),
            masks=mask_encode)

        return ann