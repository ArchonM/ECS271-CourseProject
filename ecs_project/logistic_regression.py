import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Link - https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43

# class GetDataset(Dataset):

#     def __init__(self, X_Train, Y_Train, transform=None):
#         self.X_Train = X_Train
#         self.Y_Train = Y_Train
#         self.transform = transform
#         self.X_Train = self.X_Train.transpose((0, 2, 3, 1))

#     def __len__(self):
#         return len(self.X_Train)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         x = self.X_Train[idx]
#         y = self.Y_Train[idx]

#         if self.transform:
#             x = self.transform(x)

#         return x, y

class GetDataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length
    
X_train = pd.read_pickle('../../ecsTest/ecs_project/X_train.pkl')
y_train = pd.read_pickle('../../ecsTest/ecs_project/y_train.pkl')
X_test = pd.read_pickle('../../ecsTest/ecs_project/X_test.pkl')
y_test = pd.read_pickle('../../ecsTest/ecs_project/y_test.pkl')

# testdataset = GetDataset(X_test, y_test, transform=transforms.Compose([transforms.ToTensor()]))
testdataset = GetDataset(X_test, y_test)

test_data_loader = DataLoader(testdataset, batch_size=64, shuffle=True, num_workers=0)

# Just for trial
dataiter_test = iter(test_data_loader)
images_test, labels_test  = next(dataiter_test)
print(len(images_test))
print(images_test[1].shape)
print(len(labels_test))

# traindataset = GetDataset(X_train, y_train, transform=transforms.Compose([transforms.ToTensor()]))
traindataset = GetDataset(X_train, y_train)

train_data_loader = DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=0)

# Just for trial

dataiter_train = iter(train_data_loader)
images_train, labels_train = next(dataiter_train)
print(len(images_train))
print(images_train[1].shape)
print(len(labels_train))

classes = (0, 1)

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(100*100*2,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = x.view(-1, 100*100*2)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x



learning_rate = 0.001
epochs = 10
# Model , Optimizer, Loss
model = Net()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()

#forward loop
losses = []
accur = []
for i in range(epochs):
  for j,(x_train,y_train) in enumerate(train_data_loader):
    
    #calculate output
    output = model(x_train)
 
    #calculate loss
    loss = loss_fn(output,y_train.reshape(-1,1))
 
    #accuracy
    predicted = model(torch.tensor(X_test,dtype=torch.float32))
    acc = (predicted.reshape(-1).detach().numpy().round() == y_test).mean()
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i%50 == 0:
    losses.append(loss)
    accur.append(acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))
    
# plt.plot(losses)
# plt.title('Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('loss')

# plt.plot(accur)
# plt.title('Accuracy vs Epochs')
# plt.xlabel('Accuracy')
# plt.ylabel('loss')

with open('../../ecsTest/ecs_project/result_10_epochs_logistic_r.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(losses,accur))
    
print("Done!!")