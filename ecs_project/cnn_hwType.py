import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


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
    
X_train = pd.read_pickle('../../ecsTest/ecs_project/X_train.pkl')
y_train = pd.read_pickle('../../ecsTest/ecs_project/y_train.pkl')
X_test = pd.read_pickle('../../ecsTest/ecs_project/X_test.pkl')
y_test = pd.read_pickle('../../ecsTest/ecs_project/y_test.pkl')

testdataset = GetDataset(X_test, y_test, transform=transforms.Compose([transforms.ToTensor()]))

test_data_loader = DataLoader(testdataset, batch_size=8, shuffle=True, num_workers=0)

dataiter = iter(test_data_loader)
images, labels  = next(dataiter)
print(len(images))
print(images[1].shape)