import os
import argparse
    
parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--pos_mask_thr', default=0.35, type=float)
parser.add_argument('--neg_mask_thr', default=0.80, type=float)
parser.add_argument('--num_mask_point_gt', default=10,type=int)
parser.add_argument('--box_cam_thr', default=0.2, type=float)
parser.add_argument('--work_dir', default='work_dir-proj', type=str)
parser.add_argument('--rec_weight', default=1.0, type=float)
parser.add_argument('--step1', default=8, type=int)
parser.add_argument('--step2', default=11, type=int)
parser.add_argument('--max_epochs', default=12, type=int)
parser.add_argument('--corr_size', default=21, type=int)
parser.add_argument('--scale_factor', default=2, type=int)
parser.add_argument('--obj_tau', default=0.90, type=float)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--num_semantic_points', default=5, type=int)
parser.add_argument('--epoch_semantic_centers', default=0, type=int)
parser.add_argument('--semantic_to_token', action='store_true')
parser.add_argument('--pca_dim', default=64, type=int)
parser.add_argument('--mean_shift_times_local', default=10, type=int)
parser.add_argument('--offset_range', default=5, type=int)
args, unknown = parser.parse_known_args()


if __name__ == "__main__":
    # config = 'configs/mae/attnshift_align_teacher_student.py'
    # config = 'configs/mae/attnshift_deform_attn_norm_fpn.py'
    config = 'configs/mae/attnshift_deform_attn_dense_reppoints.py'
    # config = 'configs/mae/imted_small_faster_rcnn_pointsup_pointmasksample_reconstruct_cos_voc12aug_1x_align.py'
    # config = 'configs/mae/imted_small_faster_rcnn_pointsup_pointmasksample_reconstruct_cos_voc12aug_1x.py'
    
    config_name = os.path.basename(config).split('.')[0]
    os.system(f"python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --master_port=12345 --use_env ./tools/train.py \
                {config} --cfg-options \
                    model.backbone.use_checkpoint=True \
                    model.roi_head.bbox_head.seed_score_thr=0.05 \
                    model.roi_head.bbox_head.cam_layer=7 \
                    model.roi_head.mil_head.num_layers_query=7 \
                    model.roi_head.bbox_head.seed_thr={args.box_cam_thr} \
                    model.roi_head.bbox_head.seed_multiple=0.5 \
                    model.roi_head.bbox_head.rec_weight={args.rec_weight} \
                    model.pos_mask_thr={args.pos_mask_thr} \
                    model.neg_mask_thr={args.neg_mask_thr} \
                    model.num_mask_point_gt={args.num_mask_point_gt} \
                    model.corr_size={args.corr_size} \
                    model.obj_tau={args.obj_tau} \
                    model.roi_head.mask_head.scale_factor={args.scale_factor} \
                    data.samples_per_gpu=2 \
                    data.workers_per_gpu=2 \
                    model.roi_head.num_semantic_points={args.num_semantic_points} \
                    model.roi_head.epoch_semantic_centers={args.epoch_semantic_centers} \
                    model.roi_head.semantic_to_token={args.semantic_to_token} \
                    model.roi_head.mean_shift_times_local={args.mean_shift_times_local} \
                    model.roi_head.pca_dim={args.pca_dim} \
                    optimizer_config.update_interval=2 \
                    lr_config.step=[{args.step1},{args.step2}] \
                    runner.max_epochs={args.max_epochs} \
                --work-dir {args.work_dir} \
                --gpus {args.num_gpus} --launcher pytorch")