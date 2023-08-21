#!/usr/bin/env bash

# setup

pip install numba Cython \
&& pip install ninja yacs cython matplotlib tqdm opencv-python tensorboardX einops scipy \
&& pip install mmcv-full==1.3.8 \
&& pip install timm chainercv einops 

cd apex-master
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd Connected_components_PyTorch
python3 setup.py install
python3 setup.py test
