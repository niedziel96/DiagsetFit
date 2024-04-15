import torch
import pandas as pd
import numpy as np
from torch import nn
import copy 

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, transformer, num_class, inp):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = copy.deepcopy(transformer)
        self.classifier = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, num_class))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

def build_dinov2(model_type, req_grad, num_class):
    if model_type == 'dinov2_small':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = DinoVisionTransformerClassifier(dinov2_vits14, num_class, 384)
    elif model_type == 'dinov2_base':
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model = DinoVisionTransformerClassifier(dinov2_vitb14, num_class, 768)
    elif model_type == 'dinov2_large':
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        model = DinoVisionTransformerClassifier(dinov2_vitl14, num_class, 1024)
    else: 
        print('Unknown model')

    if req_grad:
        for param in model.parameters():
            param.requires_grad = True
    
    return model 