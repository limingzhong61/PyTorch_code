import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
"""
① Prepare dataset
# Dataset and Dataloader
"""


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        print(xy.shape)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../Multiple_Dimension_Input_7\\diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)  # num_workers 多线程

"""
②Design model using Class
inherit from nn.Module
"""


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
'''
③Construct loss and optimizer
using PyTorch API
'''
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
'''
④Training cycle
forward, backward, update
'''
mse_list = []
epoch_mse_list = []
if __name__ == '__main__':
    # for epoch in range(100):
    # mini-scale
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            mse_list.append(loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
        epoch_mse_list.append(sum(mse_list) / len(mse_list))
        mse_list.clear()
# epoch-mse graph
    plt.plot(range(len(epoch_mse_list)), epoch_mse_list)
    plt.ylabel('Lost')
    plt.xlabel('epoch')
    plt.show()
## 生成test结果
x_test = dataset.x_data
y_test = model(x_test)

print('y_pred = ', y_test.data)