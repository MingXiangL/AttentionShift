import os
import argparse
    

if __name__ == "__main__":
    config = 'configs/mae/attnshift_voc12aug.py'
    
    config_name = os.path.basename(config).split('.')[0]
    os.system(f"python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 --use_env ./tools/train.py \
              {config} --cfg-options \
                data.samples_per_gpu=1 \
                data.workers_per_gpu=2 \
                optimizer_config.update_interval=2 \
              --gpus 8 --launcher pytorch")