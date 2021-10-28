import operator
import numpy as np


class KNN:
    def __init__(self, X_test, X_train, y_train, k=3, p=2):
        """
        parameter: k 临近点个数
        parameter: p 距离度量
        """
        self.k = k
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def distance(self, x, y):
        """

        :param x: 单个测试样本
        :param y: 训练集X_train
        :return: 单个样本到训练集上各点的距离
        """
        X_trainSize = self.X_train.shape[0]  # 获得第一个维度的长度
        # 将x重复dataSetSize次使其shape和X_train一样
        x = np.tile(x, (X_trainSize, 1))
        if len(x) == len(y) and len(x) > 1:
            sum = np.zeros((len(x)))
            # 计算距离，p=2为欧式距离
            for i in range(len(x)):
                sum[i] = np.sum(np.power(x[i] - y[i], self.p))
            sum = np.power(sum, 1 / self.p)
            # print(sum)
            return sum
        else:
            print("illegal data!")

    def predictData(self, inX):  # inX 为单个测试数据

        distances = self.distance(inX, self.X_train)
        # argsort函数返回的是distances值从小到大的--索引值
        sortedDistIndicies = distances.argsort()
        # 定义一个记录类别次数的字典
        classCount = {}
        # 选择距离最小的k个点
        for i in range(self.k):
            # 取出前k个元素的类别
            voteIlabel = self.y_train[sortedDistIndicies[i]]
            # 字典的get()方法，返回指定键的值，如果值不在字典中返回0
            # 计算类别次数
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # python3中用items()替换python2中的iteritems()
        # key = operator.itemgetter(1)根据字典的值进行排序
        # key = operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        # 返回次数最多的类别，即所要分类的类别
        return sortedClassCount[0][0]

    def kNN(self):
        predict = []
        for i in range(self.X_test.shape[0]):
            predict.append(self.predictData(self.X_test[i]))
        return np.array(predict)
