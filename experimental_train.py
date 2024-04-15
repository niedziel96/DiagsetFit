import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import json
import os
from os.path import exists
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

class EarlyStopper:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import pickle

#Organizing the dataset
# data_dir = '../input/dataset-breks/data set'
train_dir = '/media/admingpu/Crucial X6/BH_final_dataset/all_class_under/40x/train'
valid_dir = '/media/admingpu/Crucial X6/BH_final_dataset/all_class_under/40x/test'
batch_size = 16
use_gpu = torch.cuda.is_available()

data_transforms = {
    'train': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# def npy_loader(path):
#     sample = torch.from_numpy(np.load(path))
#     sample = sample.permute(2,0,1)
#     return sample

image_datasets = {}


image_datasets['train'] = train_datasets = datasets.ImageFolder(
    root=train_dir,
    # loader=npy_loader,
    # extensions=['.npy'],
    transform=data_transforms['train']
)

image_datasets['valid'] = valid_datasets = datasets.ImageFolder(
    root=valid_dir,
    # loader=npy_loader,
    # extensions=['.npy'],
    transform=data_transforms['valid']
)

#Charger les jeux de données avec ImageFolder
# image_datasets['train'] = train_datasets = datasets.ImageFolder(train_dir,data_transforms['train'])
# image_datasets['valid'] = valid_datasets = datasets.ImageFolder(valid_dir,data_transforms['valid'])
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size,shuffle=True, num_workers=4)
dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['valid'] = valid_loader
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}


model = torchvision.models.vit_l_32(weights='IMAGENET1K_V1')

num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, 8)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    early_stopper = EarlyStopper(patience=10, min_delta=0.1)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_v = 0.0
    best_acc_T = 0.0
    best_loss_v= 1.0
    best_loss_T= 1.0
    loss_dict = {'train': [], 'valid': []}
    acc_dict = {'train': [], 'valid': []}

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print('to start iter..')
            # Iterate over data.
            counter = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward() #L'appel de .backward()plusieurs fois accumule le gradient (par addition) pour chaque paramètre. C'est pourquoi vous devez appeler optimizer.zero_grad()après chaque .step()appel.
                        optimizer.step()#est effectue une mise à jour des paramètres basée sur le gradient actuel SGD

                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if counter%50 == 0: 
                    print(f'----- {sum(preds == labels.data)/16} -----')
                    print(f' -------  {counter}/{len(dataloaders[phase])}  -------')

                counter += 1

            # backward + optimize only if in training phase
            if phase == 'train':
                scheduler.step()

                #print(labels)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

           # copier en profondeur le modèle
            if phase == 'valid' :
                if epoch_acc > best_acc_v  :
                   best_acc_v = epoch_acc
                   best_model_wts = copy.deepcopy(model.state_dict())
                if best_loss_v > epoch_loss:
                   best_loss_v = epoch_loss   

            if phase == 'train' :
                if epoch_acc > best_acc_T:
                   best_acc_T = epoch_acc
                if best_loss_T > epoch_loss:
                   best_loss_T = epoch_loss    
            if phase == 'valid':
                valid_loss_main = epoch_loss
        if early_stopper.early_stop(valid_loss_main):    
            print('** early stopping activated **')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc_v))
    print('Best train  accuracy: {:4f}'.format(best_acc_T))
    print('valid losss: {:4f}'.format(best_loss_v))
    print('train losss: {:4f}'.format(best_loss_T))


    #   charger les meilleurs poids de modèle
    model.load_state_dict(best_model_wts)
    return model,loss_dict, acc_dict,time_elapsed
    
ress_loss = {'train': [], 'valid': []}
ress_acc = {'train': [], 'valid': []}
time_elapse=0

# Train a model with a pre-trained network
res_loss = {'train': [], 'valid': []}
res_acc = {'train': [], 'valid': []}
if use_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()
# NLLLoss because our output is LogSoftmax
criterion = nn.CrossEntropyLoss()
# Adam optimizer with a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0005)
# Decay LR by a factor of 0.1 every 5 epochs 15
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

model_ft,loss_dict, acc_dict,time_elapsed = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
res_loss = loss_dict
res_acc = acc_dict

#saving 
print('*** saving outputs ***')
with open('res_loss_8.pkl', 'wb') as fp:
    pickle.dump(res_loss, fp)
    print('res_loss saved successfully to file')

with open('res_acc_8.pkl', 'wb') as fp:
    pickle.dump(res_acc, fp)
    print('res_acc saved successfully to file')

torch.save(model_ft.state_dict(), "vit_8.pth")
torch.save(model_ft, "vit_entire_model_8.pth")

