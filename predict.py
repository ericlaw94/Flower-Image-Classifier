import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
import torch
import time
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
import json
import argparse



data_dir  = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'


with open('cat_to_name.json', 'r') as f:
    flower = json.load(f)

print(json.dumps(flower, sort_keys=True, indent=2, separators=(',', ': ')))



def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
   
    model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
         param.requires_grad = False
 
    model.class_to_idx = checkpoint[8]
    
    classifier = nn.Sequential(OrderedDict([('hidden1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('hidden2', nn.Linear(4096, len(flower))),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.load_state_dict(checkpoint[7])
    
    return model
    
model = load_checkpoint('checkpoint.pth')
model
 

    

    
   