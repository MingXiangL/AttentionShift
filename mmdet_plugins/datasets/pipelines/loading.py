from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

@PIPELINES.register_module()
class LoadAnnotationsC(LoadAnnotations):
    ''' add load_center, with tag points
    '''
    def __init__(self,
                 with_bbox=True,
                 with_center=False,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        super().__init__(with_bbox, with_label, with_mask, with_seg, poly2mask, file_client_args)
        self.with_center = with_center
    
    def _load_centers(self, results):
        if 'points' in results['ann_info']:
            results['gt_centers'] = results['ann_info']['points'].copy()
        return results
    
    # whether need to change is unknown.
    def _load_bboxes(self, results):
        return super()._load_bboxes(results)

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_center:
            results = self._load_centers(results)
            if results is None:
                return None
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_center={self.with_center}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str



