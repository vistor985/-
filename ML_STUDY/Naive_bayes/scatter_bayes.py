import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from collections import Counter
import math


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 处理X_train离散情况
    def summariz_scatter(self, train_data):
        data = zip(*train_data)  # 事实上zip()出来是一个迭代器只能前进不能后退，所以一次迭代后就用不了了
        feature = [set(i) for i in data]
        summa = []
        for i, feature_i in enumerate(feature):
            summa.append(  # 加的操作为拉普拉斯修正
                {fea: (list(zip(*train_data))[i].count(fea) + 1) / (len(list(zip(*train_data))[i]) + len(feature_i)) for
                 fea in feature_i})
        return summa

    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        # print("data:", data)
        # 将样本按label分类好，即label:所有属于这个label的样本组成的列表
        for f, label in zip(X, y):
            data[label].append(f)
        # print("data:", data)
        self.model = {
            label: self.summariz_scatter(value)
            for label, value in data.items()  # 取出键和值
        }
        print("model", self.model)
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities_scatter(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 0
            for i in range(len(value)):
                proba = value[i]
                probabilities[label] += proba[input_data[i]]  # 防止下溢出
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(  # 选出概率最高的类别
            self.calculate_probabilities_scatter(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))


def loadCSVfile2():
    tmp = np.loadtxt("balance-scale.csv", dtype=str, delimiter=",")
    for line in tmp:
        if line[0] == 'L':
            line[0] = 0
        elif line[0] == 'B':
            line[0] = 1
        else:
            line[0] = 2
    data = tmp[1:, 1:].astype(float)  # 加载数据部分
    label = tmp[1:, 0].astype(float)  # 加载类别标签部分
    # print(tmp)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    return x_train, x_test, y_train, y_test  # 返回array类型的数据


def main():
    # X, y = create_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # X, y = load_digits(return_X_y=True)
    # # 随机划分训练集和验证集,使用sklearn中的方法
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test, y_train, y_test = loadCSVfile2()
    model = NaiveBayes()
    model.fit(X_train, y_train)
    # print(model.predict([1, 2, 3, 4]))
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()
