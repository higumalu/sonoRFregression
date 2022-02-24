# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 01:14:13 2022

@author: higumalu
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
#device = ("cuda" if torch.cuda.is_available() else "cpu")

rf_dir_path = './fattyliver_rf/ndarray'
label_path = './dataset_CAP.npy'
model_path = './stepmodel/model_scripted.pt'
num_epochs = 100
learning_rate = 1e-4
train_CNN = False
batch_size = 1
shuffle = True
pin_memory = True
num_workers = 0

input_dim = 206, 600
output_dim = 1



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = np.load(annotations_file,allow_pickle=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        #print(img_path)
        image = np.load(img_path)
        image = torch.from_numpy(image)
        #image = image.type(torch.FloatTensor)
        image = image.type(torch.DoubleTensor)
        label = self.img_labels[idx, 1]
        #label = float(label)
        label=torch.tensor(label, dtype=torch.double) 
        '''
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        '''
        return image, label

dataset = CustomImageDataset(label_path, rf_dir_path,transform=None)
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_set, validation_set = torch.utils.data.random_split(dataset,[train_set_size, valid_set_size])
print(train_set_size)

train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
test_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)
print(train_loader)

# Create CNN Model
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
    
    
    
model = CNN_Model().double()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters
loss_func = nn.MSELoss()   # the target label is not one-hotted
input_shape = (-1,1,600,206)



def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    critical = 0.1
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1.Define variables
            train = Variable(images.view(input_shape))
            labels = Variable(labels)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            diff = abs(predicted - labels)
            #print(diff)
            correct_train += (diff<critical).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
            
        for images, labels in test_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs = model(test)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            diff = abs(predicted - labels)
            #print(diff)
            correct_test += (diff<critical).float().sum()
            
            #correct_test += (diff < 0.1).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
        if (epoch+1)%5 == 0:
            torch.save(model,'./stepmodel/{}Epoch'.format(epoch+1)+'.pt')
    return training_loss, training_accuracy, validation_loss, validation_accuracy


training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader)
torch.save(model,'finEpoch'+'.pt')
