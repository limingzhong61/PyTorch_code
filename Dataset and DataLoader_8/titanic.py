# 如果你没有安装pandas，请取消下一行的注释
# !pip install pandas

# %matplotlib inline
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# download('kaggle_house_train')
# train_data = pd.read_csv(download('kaggle_house_train'))
# test_data = pd.read_csv(download('kaggle_house_test'))
train_data_fname = './titanic_data/train.csv'
test_data_fname = './titanic_data/test.csv'
train_df = pd.read_csv(train_data_fname)
test_df = pd.read_csv(test_data_fname)
combine = [train_df, test_df]
# print(train_data.shape)
# print(test_data.shape)
# print(type(train_data))
# print(train_data.columns)
#
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


# --------------------prase data------------------------------
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

train_df.head()

guess_ages = np.zeros((2, 3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_df.head()

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                            ascending=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))

# -------------------------------- parse data -------------------------------------------
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
# numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# # 这个地方其实是要注意的，因为normalization应该统一用train set的均值和方差
# # 这里其实是小小的作弊了哈哈 因为测试集的均值和方差在这里被观察到了，不再是一个严格意义上的blind data了
# all_features[numeric_features] = all_features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std()))
# # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
# all_features[numeric_features] = all_features[numeric_features].fillna(0)
# all_features.shape
#
# # 因为Address特征一列有很多值，故将字符串直接映射为int值
# embed_column_names = ["Name", "Cabin", "Ticket"]
# # one_hot_column_names = ["Sex",]
# for embed_column_name in embed_column_names:
#     all_features[embed_column_name] = pd.factorize(all_features[embed_column_name])[0].astype(int)
# all_features
#
# # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# all_features = pd.get_dummies(all_features, dummy_na=True)
# all_features.shape
# all_features
#
# # 前面将训练集和测试集加在了一起，这里是再将这两个分开
# n_train = train_df.shape[0]
# train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
# test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# # Survived 是 train.csv数据集中预测结果label标签，即y^hat
# train_labels = torch.tensor(
#     train_df["Survived"].values.reshape(-1, 1), dtype=torch.float32)
# n_train, train_features.shape, test_features.shape

from torch.utils.data import Dataset, DataLoader


class TitanicDataset(Dataset):
    # ratio/10
    def __init__(self, train_features, train_labels, isTrain, ratio=8):
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


X_train = train_df.drop("Survived", axis=1)
print(X_train.info())
print(X_train.columns)
for column_name in X_train.columns:
    X_train[column_name] = X_train[column_name].astype('float32')
print(X_train.info())
X_train = X_train.values
Y_train = train_df["Survived"].astype('float32').values
# 转为二维
Y_train = Y_train.reshape(-1, 1)
X_test = test_df.drop("PassengerId", axis=1).copy().astype('float32').values
print(X_train.shape, Y_train.shape, X_test.shape)
print(type(X_train))
train_dataset = TitanicDataset(X_train, Y_train, isTrain=True, ratio=10)
# print("train_dataset.x_data[:10]", train_dataset.x_data[:10])
# print("train_dataset.y_data[:10]", train_dataset.y_data[:10])
test_dataset = TitanicDataset(X_train, Y_train, isTrain=False, ratio=8)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)  # num_workers 多线程
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)  # num_workers 多线程


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
# print(train_features.shape)

'''
③Construct loss and optimizer
using PyTorch API
'''
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
'''
④Training cycle
forward, backward, update
'''


def train(epoch):
    total_loss = 0.0
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
        # 获得一个批次的数据和标签
        # 1. Prepare data
        inputs, target = data

        # 获得模型预测结果
        # 2. Forward
        outputs = model(inputs)
        # 交叉熵代价函数
        loss = criterion(outputs, target)
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        if batch_idx % 30 == 29:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    return total_loss / len(train_loader)


def test(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # print(outputs.data.numpy())
            # _,\
            predicted = [1 if y > 0.5 else 0 for y in outputs.data.numpy()]
            total += labels.size(0)
            for i in range(len(labels)):
                correct += (predicted[i] == labels[i])
    accuracy = (100 * correct / total).numpy()[0]
    print('accuracy on test set: %d %% ' % accuracy)
    return accuracy


if __name__ == '__main__':
    epoch_loss = []
    epoch_accuracy_list = []
    epoch_count = 1
    for epoch in range(epoch_count):
        one_loss = train(epoch)
        epoch_loss.append(one_loss)
        epoch_accuracy = test(test_loader)
        epoch_accuracy_list.append(epoch_accuracy)
    ## Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    ## Add title
    plt.title("epoch_loss")
    sns.lineplot(data=pd.Series(epoch_loss))
    plt.show()
    ## Add title
    plt.title("epoch_accuracy_list")
    sns.lineplot(data=pd.Series(epoch_accuracy_list))
    plt.show()

# 将网络应用于测试集。
test_features = torch.tensor(X_test)
pred_y = model(test_features)
# 广播机制
pred_y = (pred_y > 0.5).reshape(-1).numpy() + 0
print(pred_y.shape)
# 将其重新格式化以导出到Kaggle
test_df['Survived'] = pd.Series(pred_y)
submission = pd.concat([test_df['PassengerId'], test_df['Survived']], axis=1)
submission.to_csv('submission.csv', index=False)
