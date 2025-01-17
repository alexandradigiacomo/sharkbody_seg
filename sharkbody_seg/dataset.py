import yaml
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import crop
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import numpy as np

def sharkBodyCrop(image, mask, crop_size=None):
    """
    Crops the input img to a random pixel within the shark
    Args:
        image np.array(): Drone image anywhere in between 1500,3000 and
            3000,4000.  
        mask np.array(): Binary mask with 1 for shark and 0 for no shark.
            Assumed to have the same number of pixels as the image.
        cropSize int: If None, does not apply any crop
    Returns:
        image_cropped
        mask_cropped
    """
    if not crop_size: # no cropping, return original img/mask
        return image, mask

    mask_np = np.array(mask) # confirm mask is np array
    shark_pixels = np.argwhere(mask_np == 1) # get all pixels of mask
    rand_shark_pixel = shark_pixels[np.random.choice(len(shark_pixels))] # pick a random pixel in mask
    
    c_y, c_x = rand_shark_pixel # center coordinates of crop
    c_top, c_bottom = c_y - crop_size/2, c_y + crop_size/2
    c_left, c_right = c_x - crop_size/2, c_x + crop_size/2

    # if crop is oob
    c_top = 0 if c_top < 0 else c_top
    c_left = 0 if c_left < 0 else c_left
    c_bottom = mask_np.shape[0] if c_bottom > mask_np.shape[0] else c_bottom
    c_right = mask_np.shape[1] if c_right > mask_np.shape[1] else c_right
    # preserve crop size if oob
    c_top = c_bottom - crop_size if c_bottom - c_top < crop_size else c_top
    c_left = c_right - crop_size if c_right - c_left < crop_size else c_left

    #crop img, mask
    image_cropped = image.crop((c_left, c_top, c_right, c_bottom))
    mask_cropped = mask.crop((c_left, c_top, c_right, c_bottom))

    return image_cropped, mask_cropped

class SharkBody(Dataset):

    def __init__(self, cfg, split='train'): # collect and index dataset inputs/labels
        self.cfg = cfg # store config file
        self.data_root = cfg['data_root'] # root folder
        self.annotations_root = cfg['annotations_root'] # annotations root folder
        self.split = split # determine split

        # Transforms
        transform_list = []
        transform_list.extend([Resize((cfg['image_size'], cfg['image_size'])), ToTensor()]) # add normal transforms - mo sharkbodycrop here

        self.transform = Compose(transform_list) # final transform list
        
        annotations_path = os.path.join( # path to annotations
            self.annotations_root,
            'train.json' if self.split == 'train' else 'val.json') # choose one of two splits

        self.coco = COCO(annotations_path) 
        self.data = [] # for storing masks
        self.annotation_ids = self.coco.getAnnIds(catIds = [1])
        self.annotations = self.coco.loadAnns(self.annotation_ids)
        
        for ann in self.annotations: # loop through annotations and store
            image_id = ann['image_id']
            file_name = self.coco.loadImgs(image_id)[0]['file_name']
            binary_mask = self.coco.annToMask(ann)  
            
            self.data.append({ # store image and mask data
                'image_id': image_id,
                'file_name': file_name,
                'mask': Image.fromarray(binary_mask)
            })

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        image_name = self.data[idx]['file_name'] # pull file name
        image_path = os.path.join(self.data_root, 'images', image_name) # pull image path
        img = Image.open(image_path).convert('RGB')  # open image
        mask = self.data[idx]['mask'] # pull mask
        
        # transform: see lines 31ff above where we define our transformations
        img_cropped, mask_cropped = sharkBodyCrop(img, mask, crop_size=self.cfg['crop_size'])
        img_tensor = self.transform(img_cropped)
        mask_tensor = self.transform(mask_cropped)

        sample = dict(image=img_tensor, mask=mask_tensor*255, filename = image_name)

        return sample