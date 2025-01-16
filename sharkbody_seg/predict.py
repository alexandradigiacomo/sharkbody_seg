import os
import torch
import segmentation_models_pytorch as smp
import yaml
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from sharkbody_seg.dataset import SharkBody

# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    return parser.parse_args()

# Load the config file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Model inference
def predict(config, checkpoint_path, dataset, device):
    # Extract values from the config
    image_size = config['image_size']
    in_channels = config['in_channels']
    out_channels = config['out_channels']
    encoder_name = config['encoder_name']
    encoder_weights = config['encoder_weights']

    # Load model
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        out_channels=out_channels,
        activation='sigmoid'
    )

    # Load model checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device) # load model to correct device
    model.eval()

# Prediction setup
    predictions = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run predictions on validation set
    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            img = sample['image'].to(device)
            output = model(img)
            output = torch.sigmoid(output).cpu().numpy()
            mask = (output > 0.5).astype(np.uint8)  # Convert to binary mask

            # Access the filename 
            filename = dataset.data[idx]['file_name']

            # Append the result
            predictions.append((filename, mask.squeeze()))

    return predictions

# Save the predictions
def save_predictions(predictions, save_path):
    predictions_dir = os.path.join(os.path.dirname(os.path.dirname(save_path)), 'predictions')
    os.makedirs(predictions_dir, exist_ok=True) # create dir if doesn't exist
    for filename, mask in predictions:
        mask_image = Image.fromarray(mask) # in 0,1 format
        mask_image.save(os.path.join(predictions_dir, f"pred_{filename}"))

# Main
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint

    # Extract val data from SharkBody
    dataset = SharkBody(cfg=config, split='val') 

    # Check GPU avail and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run prediction
    predictions = predict(config, checkpoint_path, dataset, device)

    # Save predictions
    save_predictions(predictions, checkpoint_path)  # Save masks to 'predictions' folder
    print("Predictions saved successfully")
