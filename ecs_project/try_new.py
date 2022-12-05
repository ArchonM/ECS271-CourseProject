# Link - https://www.kaggle.com/code/unstructuredrahul/deep-learning-pytorch-binary-classification

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv


class GetDataset(Dataset):

    def __init__(self, X_Train, Y_Train, transform=None):
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.transform = transform
        self.X_Train = self.X_Train.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# class GetDataset(Dataset):
#   def __init__(self,x,y):
#     self.x = torch.tensor(x,dtype=torch.float32)
#     self.y = torch.tensor(y,dtype=torch.float32)
#     self.length = self.x.shape[0]
 
#   def __getitem__(self,idx):
#     return self.x[idx],self.y[idx]
#   def __len__(self):
#     return self.length

X_train = pd.read_pickle('../../ecsTest/ecs_project/X_train.pkl')
y_train = pd.read_pickle('../../ecsTest/ecs_project/y_train.pkl')
X_test = pd.read_pickle('../../ecsTest/ecs_project/X_test.pkl')
y_test = pd.read_pickle('../../ecsTest/ecs_project/y_test.pkl')



X_train =  torch.from_numpy(X_train).float()
# y_train =  torch.from_numpy(y_train.values.ravel()).float()
y_train =  torch.from_numpy(y_train).float()
X_test =  torch.from_numpy(X_test).float()
# y_test =  torch.from_numpy(y_test.values.ravel()).float()
y_test =  torch.from_numpy(y_test).float()

# X_train = X_train.type(torch.LongTensor)
# y_train = y_train.type(torch.LongTensor)
# X_test = X_test.type(torch.LongTensor)
# y_test = y_test.type(torch.LongTensor)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# testdataset = GetDataset(X_test, y_test, transform=transforms.Compose([transforms.ToTensor()]))
# testdataset = GetDataset(X_test, y_test)
testdataset = TensorDataset(X_test, y_test)
test_data_loader = DataLoader(testdataset, batch_size=100, shuffle=True, num_workers=0)

# Just for trial
dataiter_test = iter(test_data_loader)
images_test, labels_test  = next(dataiter_test)
print(len(images_test))
print(images_test[1].shape)
print(len(labels_test))


# traindataset = GetDataset(X_train, y_train, transform=transforms.Compose([transforms.ToTensor()]))
# traindataset = GetDataset(X_train, y_train)
traindataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(traindataset, batch_size=100, shuffle=True, num_workers=0)

# Just for trial

dataiter_train = iter(train_data_loader)
images_train, labels_train = next(dataiter_train)
print(len(images_train))
print(images_train[1].shape)
print(len(labels_train))

classes = (0, 1)