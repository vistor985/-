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

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        stdev += 1
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        # print("train_data:", train_data)
        # print("*train_data:", *train_data)
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]  # 求出每个属性的均值和方差
        return summaries


    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        # print("data:", data)
        # 将样本按label分类好，即label:所有属于这个label的样本组成的列表
        for f, label in zip(X, y):
            data[label].append(f)
        # print("data:", data)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()  # 取出键和值
        }
        print("model", self.model)
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 0
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] += math.log(self.gaussian_probability(
                    input_data[i], mean, stdev))
        # print("probabilities:",probabilities)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(  # 选出概率最高的类别
            self.calculate_probabilities(X_test).items(),
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

    X, y = load_digits(return_X_y=True)
    # 随机划分训练集和验证集,使用sklearn中的方法
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train, X_test, y_train, y_test = loadCSVfile2()
    model = NaiveBayes()
    model.fit(X_train, y_train)
    # print(model.predict([1, 2, 3, 4]))
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()
