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

    
with open('cat_to_name.json', 'r') as f: 
    flower = json.load(f)
        
print(json.dumps(flower, sort_keys=True, indent=2, separators=(',', ': ')))
#-------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser (description = "Flower Image Classifier")
parser.add_argument('--image_path', type=str, default='flowers/test/11/image_03147.jpg', help='PREDICTING IMAGE PATH')



arg = parser.parse_args()

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

    if checkpoint['model'] == "Densenet":
        model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
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
    ('hidden2', nn.Linear(hidden_units, len(flower))),
    ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier


    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


model = load_checkpoint('checkpoint.pth')
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

def predict(image_path, model, topk=5):
 
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
    probs, classes = predict(image_path, model)   
    print(probs)
    print(classes)

    plt.figure(figsize = (5,12))
    
    x=plt.subplot(2,1,1)

    image = process_image(arg.image_path)
    title = flower[classes[0]]
    imshow(image,x, title=title.upper());
    flower_predic = [flower[i] for i in classes]

    plt.subplot(2,1,2)
  
    sns.barplot(x=probs, y=flower_predic)
               
    plt.xlabel("Prediction Accuracy")
    plt.ylabel("Flower species")

    plt.show()
    
    return None  
#---------------------------------------------------------------------------------------------------------------------------


probabilities, classes = predict(image_path,model,topk=5)
print(probabilities)
print(classes)
checking(image_path)
    
    