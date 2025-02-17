import os
import argparse
import logging
import torch
import yaml
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from pathlib import Path

from sharkbody_seg.dataset import SharkBody
from sharkbody_seg.utils.utils import set_all_seeds
from sharkbody_seg.utils.utils import lookup_torch_dtype

def get_args():
    parser = argparse.ArgumentParser(description='Predict on images using a trained model')
    parser.add_argument('--cfg_path', type=str, default='runs/unet_smp/default/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint file for model weights')
    return parser.parse_args()

def predict(model, dataset, device, output_dir):
    """Run inference on each image in the dataset and save masks & images"""
    model.eval()  # set model to eval mode
    with torch.no_grad(): 
        for i, sample in enumerate(dataset):
            image = sample['image'].unsqueeze(0).to(device)  # Send image to device
            
            pred = model(image) # predict for the single image
            pred = torch.sigmoid(pred) > 0.5  # sigmoid, 0.5 is hyperparameter

            img_display = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img_display = (img_display * 255).astype(np.uint8)  # Scale to uint8 [0, 255]

            pred_display = pred.squeeze(0).squeeze(0).cpu().numpy()  # convert from four dimensions to two (H, W)
            pred_display = (pred_display * 255).astype(np.uint8)  # convert to uint8 for compatability with cv2

            filename_root, _ = os.path.splitext(sample['filename'])  # grab filename w/o extension

            img_path = Path(output_dir) / f"img_{filename_root}.png" # (*** change to filename)
            pred_path = Path(output_dir) / f"pred_{filename_root}.png"

            cv2.imwrite(str(img_path), img_display)
            cv2.imwrite(str(pred_path), pred_display)
    
    logging.info(f"Predictions saved to {output_dir}")

if __name__ == '__main__':
    args = get_args() # read command line args
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # init logging
    cfg = yaml.safe_load(open(args.cfg_path, 'r')) # import cfg 
    dtype = lookup_torch_dtype(cfg['dtype'])

    # Init cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Initialize random number generator
    set_all_seeds(cfg['seed'], device=device.type, 
                  use_deterministic_algorithms=cfg['use_deterministic_algorithms'],
                  warn_only=True)

    # Load model
    if cfg['model_key'] == 'unet_smp':
        cfg['model_args'] = {
            'encoder_name': cfg['encoder_name'],
            'encoder_weights': cfg['encoder_weights'],  # Pretrained weights here, e.g. "imagenet"
            'in_channels': cfg['in_channels'],
            'classes': cfg['out_channels'],
        }
        model = smp.Unet(**cfg['model_args'])
    else:
        raise NotImplementedError(f'model_key, {cfg["model_key"]}, from config.yaml not implemented')

    model = model.to(device=device)

    # Optionally load model weights from a checkpoint if provided via command-line
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.error(f"Checkpoint file not found at {checkpoint_path}. Exiting...")
            exit(1)
    else:
        logging.info("No checkpoint provided. Initializing model with pretrained encoder weights.")

    # Load the dataset for prediction
    if not os.path.exists(cfg['path_data']):
        SharkBody.download(cfg['path_data'])
    
    val_set = SharkBody(cfg, split="val") ## this is split dependent (change for test!!!)
    output_dir = Path(cfg['path_checkpoints']) / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the predictions
    predict(model, val_set, device, output_dir)

    logging.info("Finished predictions.")
