# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:25:22 2022

@author: molan
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.io import read_image

class Brain_MRI(Dataset) :
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        #image = image.repeat(3,1,1)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
