"""
@author : Hyunwoong
@when : 8/22/2019
@homepage : https://github.com/gusdnd852
"""
import pandas as pd
import torch
from torch import nn

#from temp.config import device, data_path

device = torch.device("cpu")
#data_path = 'C:\Users\user\Desktop\데이터_최종__'
data = pd.read_csv('clustering_result.csv', encoding='utf-8')
label = data['clustering_KM'].values
data = data.drop('clustering_KM', axis=1).values
m = int(len(data) * 5 / 6)
train_data, test_data, train_label, test_label = data[:m], data[m:], label[:m], label[m:]

train_data = torch.tensor(train_data, device=device, dtype=torch.float)
test_data = torch.tensor(test_data, device=device, dtype=torch.float)
train_label = torch.tensor(train_label, device=device, dtype=torch.float)
test_label = torch.tensor(test_label, device=device, dtype=torch.float)


class Normalization(nn.Module):

    def __init__(self, e=1e-12):
        super().__init__()
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        z = x.std(-1, keepdim=True)
        return (x - u) / (z + self.e)


class ResidualBlock(nn.Module):

    def __init__(self, node, drop_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(node, node)
        self.batch_norm = nn.BatchNorm1d(node)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def residual(self, x, _x):
        return x + _x

    def forward(self, x):
        _x = x
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.residual(x, _x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class NeuralNet(nn.Module):

    def __init__(self, input_node, hidden_layer, hidden_node, drop_prob=0.1):
        super(NeuralNet, self).__init__()

        self.input_layer = nn.Sequential(
            Normalization(),
            nn.Linear(input_node, hidden_node),
            nn.BatchNorm1d(hidden_node),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_prob))

        self.blocks = nn.ModuleList([ResidualBlock(hidden_node)
                                     for _ in range(hidden_layer)])

        self.hidden_layer = nn.Sequential(*self.blocks)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_node, 7),  # multi label classification의 경우 1 대신 분류할 라벨의 수를 적어주세요
            nn.Softmax())  # multi label classification의 경우 nn.Softmax()를 사용하세요

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


input_node = train_data.size()[1]
net = NeuralNet(input_node=input_node,
                hidden_layer=12,  # 데이터에 맞게 조절하세요
                hidden_node=512)  # 데이터에 맞게 조절하세요

net.to(device)
L2_regularization = 0.001  # 정규화 (Lasso)
opt = torch.optim.Adam(params=net.parameters(), lr=1e-5, weight_decay=0)
criterion = nn.CrossEntropyLoss()  # multi label classification의 경우 nn.CrossEntropyLoss() 를 사용하세요.
epochs = 1000  # 데이터에 맞게 조절하세요

net.train()
for i in range(epochs + 1):
    x = train_data.to(device)
    y = train_label.to(device)
    y = torch.tensor(y, device=device, dtype=torch.int64)
    y_ = net.forward(x)

    opt.zero_grad()
    error = criterion(y_, y)
    error.backward()
    opt.step()
    err = error.item()

    if i % 50 == 0:
        print('step : {0} , error : {1}'.format(i, round(err, 5)))



net.eval()

acc, tot = 0, 0

x = test_data.to(device).float()

y = test_label.to(device).float()

y_ = net(x).float()


#nparray = tensor.numpy()
pred = [round(i.item()) for i in y_]

pred = torch.tensor(pred)

for i in zip(pred, y):

    if i[0].item() == i[1].item():

        acc += 1



acc /= test_data.size()[0]

print('accuracy is {0}'.format(acc))