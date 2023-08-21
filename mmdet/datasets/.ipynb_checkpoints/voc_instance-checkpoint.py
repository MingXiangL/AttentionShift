from collections import OrderedDict
from tkinter.messagebox import NO
import os.path as osp
from pycocotools import mask
import pdb
from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from mmdet.core import eval_map_segm
from mmdet.datasets import pipelines
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.xml_style import XMLDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label
import numpy as np

@DATASETS.register_module()
class VOCDatasetInstance(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall', 'mAP_Segm']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info_test(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.CLASSES
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'mAP_Segm':
            det_results = [res[0] for res in results]
            segm_results= [res[1] for res in results]
            det_eval_result = self.evaluate(det_results, metric='mAP', logger=logger, 
                        proposal_nums=proposal_nums, iou_thr=iou_thr,
                        scale_ranges=scale_ranges)
            segm_map, _ = eval_map_segm(det_results=det_results, 
                    segm_results=segm_results, 
                    annotations=annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc',
                    logger=logger)
            eval_results['mAP_Segm'] = segm_map
            eval_results.update(det_eval_result)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        
        return eval_results
    
    def get_ann_info(self, idx):
        # ann0 = super().get_ann_info(idx)
        img_id = self.data_infos[idx]['id']
        label_path = osp.join(self.img_prefix, 'SegmentationClass', f'{img_id}.png')
        inst_path= osp.join(self.img_prefix, 'SegmentationObject', f'{img_id}.png')
        label_img = read_label(label_path, dtype=np.int32)
        label_img[label_img == 255] = -1
        inst_img = read_label(inst_path, dtype=np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        mask_bin, labels = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        mask_bin = np.asfortranarray(mask_bin.transpose(1,2,0).astype(np.uint8))
        mask_encode = mask.encode(mask_bin)
        bbox_mask   = mask.toBbox(mask_encode)
        bbox_mask[:, 2:] += bbox_mask[:, :2]
        ann = dict(
            bboxes=bbox_mask.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=np.zeros((0, 4)).astype(np.float32),
            labels_ignore=np.zeros((0, )).astype(np.int64),
            masks=mask_encode)
        return ann

    def get_ann_info_test(self, idx):
        # ann0 = super().get_ann_info(idx)
        img_id = self.data_infos[idx]['id']
        label_path = osp.join(self.img_prefix, 'SegmentationClass', f'{img_id}.png')
        inst_path= osp.join(self.img_prefix, 'SegmentationObject', f'{img_id}.png')
        label_img = read_label(label_path, dtype=np.int32)
        label_img[label_img == 255] = -1
        inst_img = read_label(inst_path, dtype=np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        mask_bin, labels = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        mask_bin_f = np.asfortranarray(mask_bin.transpose(1,2,0).astype(np.uint8))
        mask_encode = mask.encode(mask_bin_f)
        bbox_mask   = mask.toBbox(mask_encode)
        bbox_mask[:, 2:] += bbox_mask[:, :2]
        ann = dict(
            bboxes=bbox_mask.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=np.zeros((0, 4)).astype(np.float32),
            labels_ignore=np.zeros((0, )).astype(np.int64),
            masks=mask_bin)
        return ann
    
