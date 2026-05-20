# sharkbody_seg 
This repository contains the processing scripts associated with DiGiacomo et al., 2026 Ontogenetic shifts in ecology and morphology of eastern Pacific white sharks revealed by computer vision (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0348174).

sharkbody_seg is a pipeline for segmenting the body of white sharks in aerial imagery using U-Net segmentation and extracting morphometric measurements total length and body spans. This repository is built on top of [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) and has been adapted specifically for white shark measurement applications.  

## Dataset 
Original dataset is available at Stanford Data Repository: https://doi.org/10.25740/tr054mz8990. 

Folder dataset/ contains original imagery, labels, and metadata
Folder runs/ contains model configurations, runs, and checkpoint files

## Repository Features
- **White Shark Body Segmentation:** Automatically generates pixel-wise masks of white shark bodies from aerial imagery.  
- **Morphometric Parameter Extraction:** Computes total length and body span measurements directly from segmentation masks.  
- **PyTorch-based:** Leverages modern deep learning architectures for semantic segmentation.  
- **Flexible and Extensible:** Can be adapted for other marine megafauna or aerial imaging datasets.  

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
