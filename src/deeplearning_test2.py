"""
@author : Hyunwoong
@when : 8/22/2019
@homepage : https://github.com/gusdnd852
"""
import pandas as pd
import torch
from torch import nn

#from temp.config import data_path, device
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
device = torch.device("cpu")

data = pd.read_csv('clustering_result.csv', encoding='utf-8')
label = data['clustering_KM'].values
data = data.drop('clustering_KM', axis=1).values
m = int(len(data) * 5 / 6)
train_data, test_data, train_label, test_label = data[:m], data[m:], label[:m], label[m:]


def get_acc(y, y_):
    acc = 0
    for i in zip(y, y_):
        if i[0] == i[1]:
            acc += 1
    return acc / len(y)


def train_test(model):
    model.fit(train_data, train_label)
    res = model.predict(test_data)
    acc = get_acc(test_label, res)
    #print(model, acc, '\n')
    print(acc)

print("SVC")
train_test(SVC())
print("random")
train_test(RandomForestClassifier())
print("KN")
train_test(KNeighborsClassifier())
#train_test(RandomForestClassifier())
print("DC")
train_test(DecisionTreeClassifier())