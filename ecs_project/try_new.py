# Link - https://www.kaggle.com/code/unstructuredrahul/deep-learning-pytorch-binary-classification

import pandas as pd 

X_train = pd.read_pickle('../../ecsTest/ecs_project/X_train.pkl')
y_train = pd.read_pickle('../../ecsTest/ecs_project/y_train.pkl')
X_test = pd.read_pickle('../../ecsTest/ecs_project/X_test.pkl')
y_test = pd.read_pickle('../../ecsTest/ecs_project/y_test.pkl')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

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