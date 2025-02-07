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

def sharkBodyCrop(image, mask, crop_size=None, is_centered=True):
    """
    Crops the input img to a random pixel within the shark
    Args:
        image np.array(): Drone image anywhere in between 1500,3000 and
            3000,4000.  
        mask np.array(): Binary mask with 1 for shark and 0 for no shark.
            Assumed to have the same number of pixels as the image.
        cropSize int: If None, does not apply any crop
    Returns:
        image_cropped, mask_cropped
    """
    if crop_size == 0: # no cropping, return original img/mask
        return image, mask, 0
    
    mask_np = np.array(mask) # confirm mask is np array
    shark_pixels = np.argwhere(mask_np == 1) # get all pixels of mask

    if is_centered: # pull in annotated center point
        if center_y is None or center_x is None:
            raise ValueError("Center coordinates must be provided when is_centered=True.")
        c_y, c_x = center_y, center_x # center coordinates of crop

    if is_centered is False: 
        rand_shark_pixel = shark_pixels[np.random.choice(len(shark_pixels))] # pick a random pixel in mask
        c_y, c_x = rand_shark_pixel # random coordinates of crop

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

    return image_cropped, mask_cropped, crop_size

def compute_custom_crop_size(relative_altitude, img_width, base_crop_size=896):
    """
    Custom crop size based on relative altitude and image width
    Args:
        relative_altitude (float): The relative altitude of the image (in m).
        img_width (int): The width of the image in pixels.
        base_crop_size (int): The base crop size (default is 896).

    Returns:
        crop_size (int): The calculated crop size in pixels.
    """
    if 0 <= relative_altitude <= 30: # Low altitudes
        if img_width <= 3000: crop_size = 672
        elif 3000 < img_width <= 4000: crop_size = 672
        else: crop_size = 896 # img_width > 4000
    
    elif 30 < relative_altitude <= 50: # Medium altitudes
        if img_width <= 3000: crop_size = 448
        elif 3000 < img_width <= 4000: crop_size = 448
        else: crop_size = 672 # img_width > 4000
    
    elif 50 < relative_altitude <= 100: # High altitudes
        if img_width <= 3000: crop_size = 448
        elif 3000 < img_width <= 4000: crop_size = 448
        else: crop_size = 672 # img_width > 4000
    
    else: crop_size = base_crop_size # otherwise, base crop size
        
    return crop_size


class SharkBody(Dataset):

    def __init__(self, cfg, split='train'): # collect and index dataset inputs/labels
        self.cfg = cfg # store config file
        self.data_root = cfg['data_root'] # root folder
        self.annotations_root = cfg['annotations_root'] # annotations root folder
        self.split = split # determine split

        # Transforms
        transform_list = []
        transform_list.extend([Resize((cfg['image_size'], cfg['image_size'])), ToTensor()]) 

        self.transform = Compose(transform_list) # final transform list
        
        annotations_path = os.path.join( # path to annotations
            self.annotations_root,
            'train.json' if self.split == 'train' else 'val.json') # choose one of two splits

        self.coco = COCO(annotations_path) 
        self.data = [] # for storing masks
        self.annotation_ids = self.coco.getAnnIds(catIds = [1])
        self.annotations = self.coco.loadAnns(self.annotation_ids)

        # load metadata and map altitude to image id
        self.image_metadata = {img['id']: img['relative_altitude'] for img in self.coco.loadImgs(self.coco.getImgIds())}

        # load shark center points
        centerpoints_csv = os.path.join(self.annotations_root, 'centerpoints.csv')
        self.centerpoints = pd.read_csv(centerpoints_csv)  # make dataframe self attribute
        self.center_dict = {row['filename']: (row['center_y'], row['center_x']) for _, row in self.centerpoints.iterrows()} # make dict


        for ann in self.annotations: # loop through annotations and store
            image_id = ann['image_id']
            file_name = self.coco.loadImgs(image_id)[0]['file_name']
            binary_mask = self.coco.annToMask(ann)  
            relative_altitude = self.image_metadata.get(image_id, 0)

            self.data.append({ # store image and mask data
                'image_id': image_id,
                'file_name': file_name,
                'mask': Image.fromarray(binary_mask),
                'relative_altitude': relative_altitude 
            })

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image_name = self.data[idx]['file_name'] # pull file name
        relative_altitude = self.data[idx].get('relative_altitude', 0)  # pull relative altitude for this image
        image_path = os.path.join(self.data_root, 'images', image_name) # pull image path
        img = Image.open(image_path).convert('RGB')  # open image
        mask = self.data[idx]['mask'] # pull mask
        is_centered = self.cfg['is_centered'] # is the mask centered
        if is_centered: # get center coords from csv
            center_y, center_x = self.center_dict.get(image_name, (None, None))
            if center_y is None or center_x is None:
                raise ValueError(f"Center coordinates for {image_name} not found in centerpoints.csv")
        img_width, img_height = img.size # pull size

        # cropping
        if self.cfg.get('use_custom_crop', False):  # Check if 'use_custom_crop' is True
            crop_size = compute_custom_crop_size(relative_altitude, img_width)  # Use custom crop size
        else:
            crop_size = self.cfg['crop_size']  # Use default crop size from the config

        img_cropped, mask_cropped, _ = sharkBodyCrop(img, mask, crop_size=crop_size, is_centered=is_centered)

        img_tensor = self.transform(img_cropped)
        mask_tensor = self.transform(mask_cropped)

        sample = dict(image=img_tensor, mask=mask_tensor*255, filename = image_name, crop_size = crop_size, relative_altitude = relative_altitude)

        return sample