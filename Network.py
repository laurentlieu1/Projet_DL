# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:58:42 2022

@author: molan
"""

import os
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import optim
import torch.nn as nn
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets


VGG16 = [64, 64, 'M', 128, 128, 'M', 256,256, 256,'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module) :
    def __init__(self, in_channels=3, num_classes=2) :
        
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
            )
    def forward(self, x) :
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        
        return x
    
    def create_conv_layers(self, architecture) :
        layers = []
        in_channels = self.in_channels
        
        for x in architecture : 
            if type(x) ==int : 
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x), nn.ReLU()]
                
                in_channels = x
            
            elif x=='M' :
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)

    
    
