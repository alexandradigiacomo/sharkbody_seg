# sharkbody_seg
Barebone template code for training UNet for segmentation of shark bodies from aerial imagery

## Installation
We recommend installing the project via [conda](https://docs.conda.io/en/latest/).
```
# click 'use this template' -> 'create a new repository' -> use your_repo_name
# replace 'earth_obs_seg' with 'your_repo_name'
git clone git@github.com:<username>/sharkbody_seg.git
cd sharkbody_seg
conda create -n sharkbody_seg python=3.12.8
conda activate sharkbody_seg
conda install -c conda-forge --file requirements.txt
pip install -e . # install 'sharkbody_seg' as python module
```

## Train a model
```
python sharkbody_seg/train.py --cfg_path runs/unet_smp/demo_run/config/config.yaml
```
