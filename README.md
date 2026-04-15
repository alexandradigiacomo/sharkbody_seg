# sharkbody_seg 
sharkBody_seg is a pipeline for segmenting the body of white sharks in aerial imagery using U-Net segmentation and extracting morphometric measurements total length and body spans. This repository is built on top of [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) and has been adapted specifically for white shark measurement applications.  

## Features
- **White Shark Body Segmentation:** Automatically generates pixel-wise masks of white shark bodies from aerial imagery.  
- **Kinematic Parameter Extraction:** Computes total length and body span measurements directly from the segmentation masks.  
- **PyTorch-based:** Leverages modern deep learning architectures for semantic segmentation.  
- **Flexible and Extensible:** Can be adapted for other marine species or aerial imaging datasets.  

## Dataset 
Original dataset is available at Stanford Data Repository: https://doi.org/10.25740/tr054mz8990. 

Folder dataset/ contains original imagery, labels, and metadata
Folder runs/ contains model configurations, runs, and checkpoint files


## Installation
```
git clone git@github.com: alexandradigiacomo/sharkbody_seg.git
cd sharkbody_seg
conda create -n sharkbody_seg python=3.12.8
conda activate sharkbody_seg
conda install -c conda-forge --file requirements.txt
pip install -e . # install 'sharkbody_seg' as python module
```

## Train a model
```
python sharkbody_seg/train.py --cfg_path runs/unet_smp/[choose_model]/config/config.yaml
```
