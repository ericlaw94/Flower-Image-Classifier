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

#-------------------------------------------------------------------------------------------------------------------------


    


#-------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser (description = "Flower Image Classifier")
parser.add_argument('--image_path', type=str, default='flowers/test/11/image_03147.jpg', help='PREDICTING IMAGE PATH')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'cat_to_name.json', help = 'class_to_name json file')
parser.add_argument('--GPU', help = "Option to use 'GPU' (cuda/cpu), default = cuda ", default='cuda', type = str)
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arg = parser.parse_args()

#-------------------------------------------------------------------------------------------------------------------------
with open(arg.json, 'r') as f:
    flower = json.load(f)

print(json.dumps(flower, sort_keys=True, indent=2, separators=(',', ': ')))
#-------------------------------------------------------------------------------------------------------------------------

def load_checkpoint(path):

    print('Loading model from checkpoint...')

    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(path)

    if 'hidden_units' in checkpoint:
        hidden_units = checkpoint['hidden_units']
    else:
        hidden_units = 1000

    if checkpoint['model'] == "densenet121":
        model = models.densenet121(pretrained=True)
      
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('hidden2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
    else:
        model = models.vgg19(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([ 
    ('hidden1',nn.Linear(25088,hidden_units)),
    ('relu',nn.ReLU()),
    ('dropout', nn.Dropout()),
    ('hidden2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier


    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


model = load_checkpoint(arg.checkpoint)
print(model)
print('Model Load Successfully')
#-------------------------------------------------------------------------------------------------------------------------
image_path = arg.image_path

def process_image(image_path):
 
    
    pil_image = Image.open(image_path)
    
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    left = (pil_image.width-224)/2
    bottom = (pil_image.height-224)/2
    right = left + 224
    top= bottom + 224
    
    pil_image = pil_image.crop((left, bottom, right, top))
    
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
   
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------

def predict(image_path, model, topk=arg.topk):
    
    model.to(arg.GPU)
 
    image = process_image(image_path)
    
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    prob= torch.exp(output)
    
    top_prob, top_indices = prob.topk(topk)
    
    top_prob = top_prob.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
   
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_prob, top_classes
#---------------------------------------------------------------------------------------------------------------------------
        
def checking(image_path):
    probs, classes = predict(image_path, model,topk=arg.topk)   
    print(probs)
    print(classes)
    x = flower[classes[0]]
    print('The flower species is:',x)
    
    
    
    return x 
#---------------------------------------------------------------------------------------------------------------------------


checking(image_path)
   
    