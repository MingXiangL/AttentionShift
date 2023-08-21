import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip, RandomCrop, Normalize, Pad


@PIPELINES.register_module()
class ResizeC(Resize):
    def _resize_centers(self, results):
        if 'gt_centers' in results:
            # scale_factor = np.array([w_scale, h_scale, w_scale, h_scale])
            results['gt_centers'] = results['gt_centers'] * results['scale_factor'][:2]

    def __call__(self, results):
        results = super().__call__(results)
        self._resize_centers(results)
        return results


@PIPELINES.register_module()
class RandomFlipC(RandomFlip):
    def center_flip(self, centers, img_shape, direction):
        flipped = centers.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0] = w - centers[..., 0]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1] = h - centers[..., 1]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0] = w - centers[..., 0]
            flipped[..., 1] = h - centers[..., 1]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        results = super().__call__(results)
        if results['flip'] and 'gt_centers' in results:
            # flip centers
            results['gt_centers'] = self.center_flip(results['gt_centers'],
                                              results['img_shape'],
                                              results['flip_direction'])
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class RandomCropC(RandomCrop):
    ''' do not support gt_bboxes_ignore | gt_labels_ignore!!!
        problem: 
            assert 2 objects: 1 cat head bbox -> center in head; 1 cat -> center in cat body
            but after crop, 1 cat head bbox -> remains the same; 1 cat -> only have head bbox now
            so do we need the second object center to be re calucated?  
    
    '''
    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True,
                 only_centers=False,
                 decide_only_centers=True,
                 ):
        super().__init__(crop_size, crop_type, allow_negative_crop, bbox_clip_border)
        self.only_centers = only_centers
        self.decide_only_centers = decide_only_centers
        assert self.only_centers or self.decide_only_centers

    def _crop_data(self, results, crop_size, allow_negative_crop):
        '''
            if only_center, we get gt_centers and gt_labels
            if with center and bbox, we check gt_bbox first, then check gt_center, 
                if bbox is valid but the center is filtered, we choose another center

        '''
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        if self.only_centers:
            # no bbox field !!
            assert 'gt_centers' in results
            gt_centers = results['gt_centers'] - np.array([offset_w, offset_h],
                                dtype=np.float32)
            valid_inds = (gt_centers[:, 0] > 0) & (gt_centers[:, 1] > 0) & (gt_centers[:, 0] < img_shape[1]) & (gt_centers[:, 1] < img_shape[0])
            if (not valid_inds.any() and not allow_negative_crop):
                return None
            results['gt_centers'] = gt_centers[valid_inds, :]
            results['gt_labels'] = results['gt_labels'][valid_inds]
            return results

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                
            if key == 'gt_bboxes' and 'gt_centers' in results:
                gt_centers = results['gt_centers'] - np.array([offset_w, offset_h],
                                dtype=np.float32)
                gt_centers = gt_centers[valid_inds, :]
                valid_cinds = (gt_centers[:, 0] > 0) & (gt_centers[:, 1] > 0) & (gt_centers[:, 0] < img_shape[1]) & (gt_centers[:, 1] < img_shape[0])
                if (not valid_cinds.any() and not allow_negative_crop):
                    return None
                if self.decide_only_centers:
                    results['gt_centers'] = gt_centers[valid_cinds, :]
                    results['gt_bboxes'] = results['gt_bboxes'][valid_cinds, :]
                    results['gt_labels'] = results['gt_labels'][valid_cinds, :]
                else:
                    results['gt_centers'][valid_cinds, :] = gt_centers[valid_cinds, :]
                    # results['gt_centers'][not valid_cinds, :] = get_center(results['gt_bboxes'][not valid_cinds, :], gt_centers[not valid_cinds, :])
                    raise NotImplementedError

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        

        return results


    def __call__(self, results):
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results



