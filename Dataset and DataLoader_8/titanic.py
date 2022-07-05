import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
① Prepare dataset
# Dataset and Dataloader 不太好用
# 使用pandas 读取数据集
"""
import pandas as pd
train_data = pd.read_csv('./titanic\\train.csv')
test_data = pd.read_csv('./titanic\\test.csv')
print(train_data)
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# (在每个样本中，第一个特征是ID，) 这有助于模型识别每个训练样本。 虽然这很方便，但它不携带任何用于预测的信息。 因此，在将数据提供给模型之前，(我们将其从数据集中删除)。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features)
# 1.1 数据预处理

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 这个地方其实是要注意的，因为normalization应该统一用train set的均值和方差
# 这里其实是小小的作弊了哈哈 因为测试集的均值和方差在这里被观察到了，不再是一个严格意义上的blind data了
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

# 前面将训练集和测试集加在了一起，这里是再将这两个分开
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# Survived 是 train.csv数据集中预测结果label标签，即y^hat
train_labels = torch.tensor(
    train_data.Survived.values.reshape(-1, 1), dtype=torch.float32)
print(n_train, train_data.shape)




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
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
