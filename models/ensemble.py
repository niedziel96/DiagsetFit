import torch 
from torch import nn 

class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        # self.modelC = modelC
        self.classifier = nn.Linear(8 * 2, 8)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        # x3 = self.modelC(x)
        x = torch.cat((x1, x2), dim=1)
        out = self.classifier(x)
        return out