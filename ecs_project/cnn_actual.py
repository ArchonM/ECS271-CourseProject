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


# https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification

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

# Testing
def test_model_whole_dataset(net_model, datasetloader, classes) :
  correct = 0
  total = 0
  with torch.no_grad():
    for data in datasetloader:
        images, labels = data
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = net_model(images)
        # softmax_outputs = nn.Softmax(dim=1)(outputs) # Softmax
        _, predicted = torch.max(outputs.data, 1)
        # predicted = softmax_outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  accuracy = (100 * correct / total)
  return accuracy

# Model_training
def train_model(net_model_t, learning_rate_t, trainloader_t, testloader_t, classes_t, number_of_epochs_t):
  loss_per_epoch = []
  training_accuracy_per_epoch = []
  testing_accuracy_per_epoch = []
  # net_model.to(device) # GPU not used as colab needed premium
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net_model_t.parameters(), learning_rate_t, momentum=0.9) # Q- momentum?
  for epoch in range(number_of_epochs_t):  # loop over the dataset multiple times
      running_loss = 0.0
      loss_sum = 0.0 
      for i, data in enumerate(trainloader_t, 0):
          # get the inputs
          inputs, labels = data
          # inputs = inputs.to(device)
          # labels = labels.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()
          # forward + backward + optimize
          outputs = net_model_t(inputs)
          
        #   outputs = outputs.type(torch.LongTensor)
          labels = labels.type(torch.LongTensor)
          loss = criterion(outputs, labels)
        #   loss = torch.from_numpy(loss).float()

          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          loss_sum += loss.item()
          if i % 100 == 99:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
      # print(len(trainloader_t))
      loss_per_epoch.append(loss_sum/len(trainloader_t))
    #   training_accuracy_per_epoch.append(test_model_whole_dataset(net_model_t, trainloader_t, classes_t))
      testing_accuracy_per_epoch.append(test_model_whole_dataset(net_model_t, testloader_t, classes_t))
      print("Done Epoch : ", epoch)
      print(loss_per_epoch)
      print(testing_accuracy_per_epoch)
    #   print(training_accuracy_per_epoch)
  print('Finished Training')
  return loss_per_epoch, testing_accuracy_per_epoch


# CNN
class Net(nn.Module):  # Question 1
  #We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. (Q- use cpu/gpu)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)  # (in-channel, out-channel, kernel_size) Q - how we decide number of i/p, o/p channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)  # (size of each input sample, size of each output sample)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        # self.fc1_q3 = nn.Linear(100*100*2, 110)  # (size of each input sample, size of each output sample)
        # self.fc2_q3 = nn.Linear(110, 74)
        # self.fc3_q3 = nn.Linear(74, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = x.view(-1, 100*100*2)
        # x = F.relu(self.fc1_q3(x))  # with ReLU
        # x = F.relu(self.fc2_q3(x))
        # x = self.fc3_q3(x)
        return x


net = Net()
print(net)

net_model_feed_q1 = net 
learning_rate_q1 = 0.001
no_of_epochs_q1 = 30
loss_per_epoch_q1, testing_accuracy_per_epoch_q1 = train_model(net_model_feed_q1, learning_rate_q1, train_data_loader, test_data_loader, classes, no_of_epochs_q1)
# accuracy_per_class(net_model_feed_q1, testloader_q1, classes_q1, batch_size_q1)


# plt.plot(range(0,no_of_epochs_q1) , loss_per_epoch_q1) # Step here is the number of batches after which accuracy was calculated
# plt.title('Loss vs epochs - Question1')
# plt.show()

# #Traing_Accuracy_graph
# plt.plot(range(0,no_of_epochs_q1) , training_accuracy_per_epoch_q1) # Step here is the number of batches after which accuracy was calculated
# # plt.title('Training_Accuracy vs epochs - Question1')
# # plt.show()

# #Testing_Accuracy_graph
# plt.plot(range(0,no_of_epochs_q1) , testing_accuracy_per_epoch_q1) # Step here is the number of batches after which accuracy was calculated
# # plt.title('Testing_Accuracy vs epochs - Question1')
# # plt.show()

# plt.legend(['Training_Acc', 'Testing_Acc'], loc='lower right')
# plt.title("Train_Acc & Test_acc vs epochs - Q1")
# plt.xlabel("No. of epochs")
# plt.ylabel("Accuracy")
# plt.show() 

with open('../../ecsTest/ecs_project/result_b100_e30_cnn_actual.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(range(0,no_of_epochs_q1), loss_per_epoch_q1,testing_accuracy_per_epoch_q1 ))
    
print("Done!!")