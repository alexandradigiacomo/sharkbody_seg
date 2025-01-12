import yaml
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import numpy as np

class SharkBody(Dataset):

    def __init__(self, cfg, split='train'): # collect and index dataset inputs/labels
        self.data_root = cfg['data_root'] # root folder
        self.split = split # determine split

        self.transform = Compose([ # transform: resize/convert to torch tensor and (opt.) augmentation      
            Resize((cfg['image_size'], cfg['image_size'])), ToTensor()])
        
        annotations_path = os.path.join( # path to annotations
            self.data_root, 'annotations', 
            'train.json' if self.split == 'train' else ('val.json' if self.split == 'val' else 'test.json')) # choose one of three splits

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

        print(f"Original Image size: {img.size}")  # Debugging step: print original size

        
        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)
        mask_tensor = self.transform(mask)

        sample = dict(image=img_tensor, mask=mask_tensor)

        return sample