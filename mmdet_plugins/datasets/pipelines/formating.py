from mmcv.parallel import DataContainer as DC

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class DefaultFormatBundleC(DefaultFormatBundle):
    def __call__(self, results):
        results = super().__call__(results)
        if 'gt_centers' in results:
            results['gt_centers'] = DC(to_tensor(results['gt_centers']))
        return results
    
