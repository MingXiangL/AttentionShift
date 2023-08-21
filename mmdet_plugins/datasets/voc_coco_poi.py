import numpy as np
from mmdet.datasets import CocoDataset, VOCDataset, DATASETS

@DATASETS.register_module()
class VOCCocoDatasetPoi(CocoDataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    
    def _parse_ann_info(self, img_info, ann_info):
        """
        ann_info is exactly what in json:
            [{'segmentation': [[219, 80, 219, 364, 469, 364, 469, 80]], 'area': 71000, 'iscrowd': 0, 'image_id': 8330, 'bbox': [219, 80, 250, 284], 'category_id': 9, 'id': 23943, 'ignore': 0, 'point': [332.226, 241.908]},... ]
        """
        gt_centers = []
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        self.with_bbox = True
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            if 'bbox' not in ann:
                self.with_bbox = False
            break

        if self.with_bbox:
            for i, ann in enumerate(ann_info):
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]
                if ann.get('iscrowd', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_masks_ann.append(ann.get('segmentation', None))
                    gt_centers.append(ann.get('point', None))
        
            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
                gt_centers = np.array(gt_centers, dtype=np.float32)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)
                gt_centers = np.zeros((0, 2), dtype=np.float32)

            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

            seg_map = img_info['filename'].replace('jpg', 'png')

            ann = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                points=gt_centers,
                bboxes_ignore=gt_bboxes_ignore,
                masks=gt_masks_ann,
                seg_map=seg_map)
            
            return ann
        
        else:
            for i, ann in enumerate(ann_info):
                if ann.get('ignore', False):
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                point = ann.get('point', None)
                if point is not None and len(point) == 2:
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_masks_ann.append(ann.get('segmentation', None))
                    gt_centers.append(point)

            if gt_centers:
                gt_labels = np.array(gt_labels, dtype=np.int64)
                gt_centers = np.array(gt_centers, dtype=np.float32)
            else:
                gt_labels = np.array([], dtype=np.int64)
                gt_centers = np.zeros((0, 2), dtype=np.float32)

            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            seg_map = img_info['filename'].replace('jpg', 'png')

            ann = dict(
                labels=gt_labels,
                points=gt_centers,
                bboxes_ignore=gt_bboxes_ignore,
                masks=gt_masks_ann,
                seg_map=seg_map)

            return ann


