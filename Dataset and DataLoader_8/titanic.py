# 如果你没有安装pandas，请取消下一行的注释
# !pip install pandas

# %matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# download('kaggle_house_train')
# train_data = pd.read_csv(download('kaggle_house_train'))
# test_data = pd.read_csv(download('kaggle_house_test'))
train_data_fname = './titanic\\train.csv'
test_data_fname = './titanic\\test.csv'
train_data = pd.read_csv(train_data_fname)
test_data = pd.read_csv(test_data_fname)

# print(train_data.shape)
# print(test_data.shape)
# print(type(train_data))
# print(train_data.columns)
#
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 将train_data 和test_data合并一起处理
# 注意这里删除了第一个特征ID，因为它不携带任何预测信息
# 因为预测值为第二个特征，故除去
all_features = pd.concat((train_data.iloc[:, 2:-1], test_data.iloc[:, 1:]))
all_features

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 这个地方其实是要注意的，因为normalization应该统一用train set的均值和方差
# 这里其实是小小的作弊了哈哈 因为测试集的均值和方差在这里被观察到了，不再是一个严格意义上的blind data了
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features.shape

# 因为Address特征一列有很多值，故将字符串直接映射为int值
embed_column_names = ["Name", "Cabin", "Ticket"]
# one_hot_column_names = ["Sex",]
for embed_column_name in embed_column_names:
    all_features[embed_column_name] = pd.factorize(all_features[embed_column_name])[0].astype(int)
all_features

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
all_features

# 前面将训练集和测试集加在了一起，这里是再将这两个分开
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# Survived 是 train.csv数据集中预测结果label标签，即y^hat
train_labels = torch.tensor(
    train_data["Survived"].values.reshape(-1, 1), dtype=torch.float32)
n_train, train_features.shape, test_features.shape

from torch.utils.data import Dataset, DataLoader


class TitanicDataset(Dataset):
    # ratio/10
    def __init__(self, isTrain, ratio):
        total_len = train_features.shape[0]  # shape(多少行，多少列)
        if isTrain:
            self.len = int(total_len * ratio / 10)
            self.x_data = train_features[:self.len, :]
            self.y_data = train_labels[:self.len, :]
        else:
            self.len = total_len - int(total_len * ratio / 10)
            self.x_data = train_features[self.len:, :]
            self.y_data = train_labels[self.len:, :]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = TitanicDataset(isTrain=True, ratio=10)
# print("train_dataset.x_data[:10]", train_dataset.x_data[:10])
# print("train_dataset.y_data[:10]", train_dataset.y_data[:10])
test_dataset = TitanicDataset(isTrain=False, ratio=8)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)  # num_workers 多线程
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)  # num_workers 多线程


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(15, 10)
        self.linear2 = torch.nn.Linear(10, 6)
        self.linear3 = torch.nn.Linear(6, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
# print(train_features.shape)

'''
③Construct loss and optimizer
using PyTorch API
'''
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
④Training cycle
forward, backward, update
'''

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 30 == 29:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # print(outputs.data.numpy())
            # _,\
            predicted = [0 if y > 0 else 1 for y in outputs.data.numpy()]
            total += labels.size(0)
            for i in range(len(labels)):
                correct += (predicted[i] == labels[i])
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
# mse_list = []
# epoch_mse_list = []
# if __name__ == '__main__':
#     for epoch in range(1000):
#         for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
#             # 1. Prepare data
#             inputs, labels = data
#             # 2. Forward
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels)
#
#             mse_list.append(loss.item())
#             # 3. Backward
#             optimizer.zero_grad()
#             loss.backward()
#             # 4. Update
#             optimizer.step()
#         epoch_one_loss = sum(mse_list) / len(mse_list)
#         epoch_mse_list.append(epoch_one_loss)
#         print("epoch:{},loss:{}".format(epoch, epoch_one_loss))
#
#     # epoch-mse graph
#     plt.plot(range(len(epoch_mse_list)), epoch_mse_list)
#     plt.ylabel('Lost')
#     plt.xlabel('epoch')
#     plt.show()
#
#     print("pred:---------------------")
#     # def pred():
#     ## 原来的train结果
#     x_test = train_features
#     y_test = model(x_test).data.numpy()
#     # print(y_test.shape)
#     # print(y_test[:10])
#     y_test = [0 if y > 0 else 1 for y in y_test]  # 表示 第一类的概率大于0.5的时候 取 第一类 即类别 0
#     pred_one = 0;
#     for item in y_test:
#         if item > 0:
#             pred_one += 1
#     print("pred one:{},pre zero:{}".format(pred_one, len(y_test) - pred_one))
#     print('y_pred = ', y_test[:10])
#     print('y_real = ', train_labels[:20])
#     real_one = 0
#     for item in train_labels:
#         if item > 0:
#             real_one += 1
#     print("real one:{},real zero:{}".format(real_one, len(y_test) - real_one))
#
# ## 生成test结果
# x_test = test_features
# y_test = model(x_test).data.numpy()
# print(abs(y_test[0]))
# print(y_test.shape)
# print(y_test)
# y_test = [0 if y > 0 else 1  for y in y_test]  # 表示 第一类的概率大于0.5的时候 取 第一类 即类别 0
# print('y_pred = ', y_test)

# print((np.array(res) == np.array(y)).sum()/len(y))
# def pred(train_features, test_features, train_labels, test_data,
#                    num_epochs, lr, weight_decay, batch_size):
#     net = get_net()
#     train_ls, _ = train(net, train_features, train_labels, None, None,
#                         num_epochs, lr, weight_decay, batch_size)
#     d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
#              ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
#     print(f'训练log rmse：{float(train_ls[-1]):f}')
# 将网络应用于测试集。
preds = model(test_features).detach().numpy()
# 将其重新格式化以导出到Kaggle
test_data['Survived'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['PassengerId'], test_data['Survived']], axis=1)
submission.to_csv('submission.csv', index=False)
