import pdb
import torch
import torch.nn as nn

from .standard_roi_head_mask_point_sample_rec_align_teacher_student import StandardRoIHeadMaskPointSampleRecAlignTS, idx_feats_by_coords
from ..utils import ObjectQueues, ObjectFactory, cosine_distance, cosine_distance_part
from ..builder import HEADS, build_loss

@HEADS.register_module()
class StandardRoIHeadMaskPointSampleRecAlignTSProject(StandardRoIHeadMaskPointSampleRecAlignTS):
    def __init__(
            self, 
            mil_head=None, 
            bbox_roi_extractor=None, 
            bbox_head=None, 
            mask_roi_extractor=None, 
            mask_head=None, 
            shared_head=None, 
            mae_head=None, 
            bbox_rec_head=None, 
            train_cfg=None, 
            test_cfg=None, 
            visualize=False, 
            epoch=0, 
            epoch_semantic_centers=0, 
            num_semantic_points=3, 
            semantic_to_token=False, 
            with_align=True, 
            pca_dim=128,
            mean_shift_times_local=10, 
            len_queque=100, 
            ratio_range=[0.9, 1.2], 
            appear_thresh=0.8, 
            max_retrieval_objs=5,
            feature_dim=384,
            project_head_dim=128,
            contrust_loss=None,
            ):
        super().__init__(mil_head, bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, mae_head, bbox_rec_head, train_cfg, test_cfg, visualize, epoch, epoch_semantic_centers, num_semantic_points, semantic_to_token, with_align, pca_dim, mean_shift_times_local, len_queque, ratio_range, appear_thresh, max_retrieval_objs)
        self.project_head = nn.Linear(feature_dim, project_head_dim)
        if contrust_loss is not None:
            self.contrust_loss = build_loss(contrust_loss)

    def forward_train(
            self, 
            x, 
            img_metas, 
            proposal_list, 
            gt_bboxes, 
            gt_labels, 
            gt_bboxes_ignore=None, 
            gt_masks=None, 
            vit_feat=None, 
            feats_teacher=None,
            img=None, 
            point_init=None, 
            point_cls=None, 
            point_reg=None, 
            imgs_whwh=None, 
            attns=None, 
            gt_points=None, 
            gt_points_labels=None, 
            mask_point_labels=None, 
            mask_point_coords=None, 
            semantic_centers=None, 
            semantic_centers_split=None, 
            teacher_point_tokens=None, 
            student_point_tokens=None,
            semantic_centers_feat_split=None,
        ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        num_imgs = len(img_metas)
        if self.train_cfg.get('point_assigner'):
            num_proposals = point_reg.size(1)
            imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
            # 需要先获得point预测结果（point_reg sigmoid之后直接当作绝对位置,相对位置不太好弄，没有w，h无法用decoder)
            point_assign_results = []
            for i in range(num_imgs):
                normalize_point_cc = point_reg[i].detach()
                point_coords = (gt_points[i][:, :2] + gt_points[i][:, 2:]) / 2
                point_labels = gt_points_labels[i]
                if (self.epoch >= self.epoch_semantic_centers) and (semantic_centers is not None) and self.semantic_to_token:
                    point_coords = torch.cat((point_coords, semantic_centers[i][0]), dim=0)
                    point_labels = torch.cat((point_labels, semantic_centers[i][1]), dim=0)
                assign_result = self.point_assigner.assign(
                    normalize_point_cc, point_cls[i], point_coords,
                    point_labels, img_metas[i]
                )
                point_sampling_result = self.point_sampler.sample(
                    assign_result, point_reg[i], 
                    point_coords
                )
                point_assign_results.append(point_sampling_result)
            bbox_targets = self.get_targets(
                point_assign_results, point_coords, point_labels, self.train_cfg,
                True)
            point_loss = self.loss(
                point_cls.view(-1, point_cls.size(-1)),
                point_reg.view(-1, 2),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        
        losses.update(point_loss)
        
        if self.visualize:
            self.gt_point_coords = point_coords
            self.point_sampling_result = point_sampling_result

        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, img=img)
            losses.update(bbox_results['loss_bbox'])
            
        # mask head forward and loss
        if self.with_mask:
            # gt_masks_pseudo = self.get_pseudo_gt_masks_from_point_attn()
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    mask_point_coords, mask_point_labels,
                                                    semantic_centers=semantic_centers_split,
                                                    img_metas=img_metas)
            losses.update(mask_results['loss_mask'])
            
        # mae head forward and loss
        if self.with_mae_head:
            loss_rec = self.mae_head(vit_feat, img)
            losses.update(loss_rec)
            
        if self.with_align:
            semantic_centers_teacher = idx_feats_by_coords(feats_teacher, semantic_centers_split)
            tokens_pos_teacher = []
            tokens_pos_student = []
            tokens_pos_teacher_proj = []
            tokens_pos_student_proj = []
            gt_bboxes_arrange = []
            gt_labels_arrange = []
            student_point_tokens_proj = self.project_head(student_point_tokens)
            teacher_point_tokens_proj = self.project_head(teacher_point_tokens)
            for i_img, res in enumerate(point_assign_results):
                student_token_img = student_point_tokens[i_img, res.pos_inds]
                teacher_token_img = teacher_point_tokens[i_img, res.pos_inds]
                student_token_img_proj = student_point_tokens_proj[i_img, res.pos_inds]
                teacher_token_img_proj = teacher_point_tokens_proj[i_img, res.pos_inds]
                tokens_pos_student.append(student_token_img)
                tokens_pos_teacher.append(teacher_token_img)
                tokens_pos_student_proj.append(student_token_img_proj)
                tokens_pos_teacher_proj.append(teacher_token_img_proj)
                gt_bboxes_arrange.append(gt_bboxes[i_img][res.pos_assigned_gt_inds])
                gt_labels_arrange.append(gt_labels[i_img][res.pos_assigned_gt_inds])
                # pdb.set_trace()
            # loss_align = self.align_forward_train(semantic_centers_teacher, tokens_pos, gt_bboxes_arrange, gt_labels_arrange)
            loss_align = self.align_forward_train(semantic_centers_feat_split, tokens_pos_student, tokens_pos_teacher, tokens_pos_student_proj, tokens_pos_teacher_proj, gt_bboxes_arrange, gt_labels_arrange)
            losses.update(loss_align)

        return losses


    def align_forward_train(self, semantic_centers_feat_split, tokens_pos_student, tokens_pos_teacher, tokens_pos_student_proj, tokens_pos_teacher_proj, gt_bboxes, gt_labels):
        num_inst = 0
        # corr_loss = torch.zeros(1, device=tokens_pos_student[0].device, dtype=tokens_pos_student[0].dtype)
        corr_loss = 0 * torch.cat(tokens_pos_student_proj).sum()
        feats_mb, cls_obj_num = self.object_queues.get_all_tokens()
        if torch.is_tensor(feats_mb):
            feats_mb_proj = self.project_head(feats_mb)
            feats_mb_cls = list(feats_mb_proj.split(cls_obj_num))
        else:
            feats_mb_cls = [[] for _ in range(self.bbox_head.num_classes)]
        for i_img, tokens_student, tokens_teacher, tokens_student_proj, tokens_teacher_proj in zip(range(len(tokens_pos_student)), tokens_pos_student, tokens_pos_teacher, tokens_pos_student_proj, tokens_pos_teacher_proj):
            sc_feats = semantic_centers_feat_split[i_img]
            if isinstance(sc_feats, list):
                continue

            for i_obj, token_s, token_t, token_s_p, token_t_p in zip(range(len(tokens_student)), tokens_student, tokens_teacher, tokens_student_proj, tokens_teacher_proj):
                sc_feat = sc_feats[i_obj]
                if torch.numel(sc_feat) == 0:
                    continue
                
                obj = ObjectFactory.create_one(
                    token_s[None], 
                    sc_feat, 
                    gt_bboxes[i_img][i_obj:i_obj+1], 
                    gt_labels[i_img][i_obj:i_obj+1],
                    device=token_s.device,
                )
                kobjs, masking = self.object_queues.get_similar_obj(obj, return_mask=True)
                
                if kobjs is not None and kobjs['token'].shape[0] >= 5:
                    kobj_tokens_proj = feats_mb_cls[gt_labels[i_img][i_obj]][masking] #pos samples
                    neg_idx = cls_one_hot(cls_obj_num, gt_labels[i_img][i_obj], token_s.device)
                    neg_samples = feats_mb_proj[neg_idx]
                    corr_loss += self.contrust_loss(token_s_p, kobj_tokens_proj, neg_samples)
                    # cost_token, cosine_sim = cosine_distance(obj.token, kobjs['token'])
                    # cost_parts = cosine_distance_part(obj.part_feats, kobjs['feature'])
                    # corr_loss += cost_token.mean()
                    num_inst += 1
                
                self.object_queues.append(
                    gt_labels[i_img][i_obj],
                    i_obj,
                    tokens_teacher,
                    sc_feats,
                    gt_bboxes[i_img],
                    device=token_t.device,
                )
                
                if torch.is_tensor(feats_mb_cls[gt_labels[i_img][i_obj]]):
                    ptr = (self.object_queues.queues[gt_labels[i_img][i_obj]].ptr - 1)% self.object_queues.len_queue
                    if feats_mb_cls[gt_labels[i_img][i_obj]].shape[0] == self.object_queues.len_queue:
                        feats_mb_cls[gt_labels[i_img][i_obj]][ptr:ptr+1] = token_t_p
                        if isinstance(feats_mb_cls, tuple):
                            print(f'11111111111')
                            pdb.set_trace()
                    else:
                        try:
                            feats_mb_cls[gt_labels[i_img][i_obj]] = torch.cat((feats_mb_cls[gt_labels[i_img][i_obj]], token_t_p[None]), dim=0)
                            cls_obj_num[gt_labels[i_img][i_obj]] += 1
                        except BaseException:
                            print(f'222222222222')
                            pdb.set_trace()
                else:
                    feats_mb_cls[gt_labels[i_img][i_obj]] = token_t_p[None]
                    cls_obj_num[gt_labels[i_img][i_obj]] += 1
                    if isinstance(feats_mb_cls, tuple):
                        print(f'333333333333333')
                        pdb.set_trace()
        loss_align = corr_loss / (num_inst + 1e-6)
        return dict(loss_align=loss_align)


def cls_one_hot(cls_num_list, cls_idx, device, mode='neg'):
    num_objs = sum(cls_num_list)
    obj_idx = torch.ones(num_objs, dtype=torch.long, device=device)
    pre_cls_num = sum(cls_num_list[:cls_idx])
    obj_idx = obj_idx[pre_cls_num : pre_cls_num+cls_num_list[cls_idx]]
    if mode == 'neg':
        return obj_idx
    else:
        return 1 - obj_idx