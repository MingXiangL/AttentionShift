import enum
import numpy as np
from mmdet.core.evaluation.class_names import get_classes
from terminaltables import AsciiTable
from mmcv.utils import print_log
from pycocotools import mask as maskUtils
from chainercv.evaluations import calc_detection_voc_ap, calc_instance_segmentation_voc_prec_rec

import pdb


def eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted masks obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in `FCIS`_.

    .. _`FCIS`: https://arxiv.org/abs/1611.07709

    Args:
        pred_masks (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
        gt_masks (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_instance_segmentation_voc_prec_rec(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap), 'precision': prec, 'recall': rec}


def transfer_results(det_results, segm_results, annotations):
    gt_masks = [ann['masks'] for ann in annotations]
    gt_labels= [ann['labels'] for ann in annotations]
    
    labels = []
    masks  = []
    scores = []
    for _, (det, seg) in enumerate(zip(det_results, segm_results)):
        lbl = []
        mks = []
        scr = []
        for i_cls in range(len(det)):
            if det[i_cls].shape[0] == 0:
                continue
            for i_inst, inst_seg in enumerate(seg[i_cls]):
                try:
                    mask_inst = maskUtils.decode(inst_seg).astype(np.bool_)
                except BaseException:
                    print(inst_seg)
                    pdb.set_trace()
                lbl.append(i_cls)
                mks.append(mask_inst)
                scr.append(det[i_cls][i_inst, -1])

        if len(lbl) == 0:
            labels.append(np.zeros(0, dtype=np.int32))
            masks.append(np.zeros((0, 5), dtype=np.float32))
            scores.append(np.zeros(0, dtype=np.float32))
        else:
            labels.append(np.array(lbl))
            masks.append(np.stack(mks).astype(np.bool_))
            scores.append(np.array(scr))
    
    #------------- Test Evaluation code with GT -------------#
    # pdb.set_trace()
    # scores_tests = [np.clip(g, 0, 1) for g in gt_labels]
    # return gt_masks, gt_labels, scores_tests, gt_masks, gt_labels
    
     #------------- Test Evaluation code with GT -------------#
    return masks, labels, scores, gt_masks, gt_labels


def eval_map_segm(det_results, 
                segm_results,
                annotations, 
                scale_ranges=None,
                iou_thr=0.5,
                dataset=None,
                logger=None,
                tpfp_fn=None,
                nproc=4):
    '''
    √ 1. 将结果转化为chainerCV采用的格式
    √ 2. 调用chainerCV的eval_instance_segmentation_voc函数得到结果
    3. 用GT和随机值验证, 应该一个是100, 一个是~0
    '''
    pred_masks, pred_labels, pred_scores, gt_masks, gt_labels = \
        transfer_results(det_results, segm_results, annotations)

    result = eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels,
        iou_thresh=iou_thr,
        use_07_metric=True)
    
    eval_results = [{'mAP_mask': result['map'], 
                'precision': result['precision'], 
                'recall':result['recall'], 
                'ap': result['ap']}]

    label_names = get_classes(dataset)
    # header = ['class', 'mask_recall', 'mask_ap']
    ## recall is wrong !!!
    # table_data = [header]
    # for j in range(len(label_names)):
    #     if result["recall"][j].shape[0] == 0:
    #         result["recall"][j] = 0.0
    #     else:
    #         result["recall"][j] = result["recall"][j].mean()
    #     row_data = [
    #     label_names[j],  f'{result["recall"][j]:.3f}', f'{result["ap"][j]:.3f}']
    #     table_data.append(row_data)
    header = ['class', 'mask_ap']
    table_data = [header]
    for j in range(len(label_names)):
        row_data = [
        label_names[j],  f'{result["ap"][j]:.3f}']
        table_data.append(row_data)

    table_data.append(['mAP', f'{result["map"]:.3f}'])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return result['map'], eval_results

    

        
