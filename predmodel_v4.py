# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:32:54 2022

@author: higumalu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:16:17 2022

@author: higumalu
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

pt_path = './stepmodel/90Epoch.pt'

transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((600, 206)),
    #transforms.CenterCrop(50),
    transforms.ToTensor()
    
])

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=7, stride=1, padding=0) #output_shape=(16,24,24)
        self.lkrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid() # activation
        # Max pool 1
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,12,12)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=0) #output_shape=(32,8,8)
        
        #self.cnn3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0) #output_shape=(32,8,8)
        
        # Max pool 2
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,4,4)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(1*64*295*98, 64) 
        self.fc2 = nn.Linear(64, 1) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu(out)
        # Max pool 1
        #out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu(out)
        #out = self.cnn3(out)
        #out = self.gelu(out)
        #print(out.size())
        # Max pool 2 
        #out = self.maxpool2(out)
        out = out.view(-1, out.size(0)*64*295*98)
        #out = out.view(1,-1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.lkrelu(out)
        out = self.fc2(out)
        return out
    
model = torch.load(pt_path)
model.eval()
print(model)
device = torch.device("cpu")
model.to(device)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(0,166):
    pred_rf = np.load('./fattyliver_rf/test-3/{}.npy'.format(i))
    #pred_rf = np.array([pred_rf,pred_rf])
    
    t_pred_rf = torch.from_numpy(pred_rf)
    
    inputdata = t_pred_rf.type(torch.DoubleTensor)
    
    inputdata = inputdata.view(1, 1, 600,206) 
    
    pred = model(inputdata).cpu().detach().numpy()
    
    print(pred[0][0])
    #print(pred,'---{}'.format(i))