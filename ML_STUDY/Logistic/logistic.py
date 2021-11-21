import random

from numpy import *
import pandas as pd
from sklearn.datasets import load_iris
# 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    # return 1.0 / (1 + exp(-inX))
    # 防止指数过大溢出
    if inX >= 0:
        return 1.0 / (1 + exp(-inX))
    else:
        return exp(inX) / (exp(inX) + 1)


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # mat可以将字符串转成矩阵
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    # for _ in range(100):
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次都降低alpha(learning rate)的大小
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            # 更新参数往极大似然函数上升最快的方向
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    # 打开训练集
    frTrain = open('horseColicTraining.txt')
    # 打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        lineArr.extend([1])
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随机上升梯度训练
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # 使用上升梯度训练
    # trainWeights = gradAscent(array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        lineArr.extend([1])
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    # 错误概率计算
    errorRate = (float(errorCount) / numTestVect) * 100
    print("测试集错误率为：%.2f%%" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteration the average error rate is: %f" % (numTests, errorSum / float(numTests)))


def loadIris():
    # 得到数据特征
    data = load_iris()
    # 得到数据对应的标签
    iris_target = data.target
    # 利用Pandas转化为DataFrame格式
    iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
    # 选择其类别为0和1的样本 （不包括类别为2的样本）
    iris_features_part = iris_features.iloc[:90]
    iris_target_part = iris_target[:90]
    # shuffle 是否在划分前打乱数据的顺序，默认为True。
    # random_state: 的随机种子，取值正整数，相同的随机种子，一定可以产生相同的数据序列。
    # test_size: 测试集数据所占有的比例。
    # 测试集大小为20%， 训练集大小为80%
    x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size=0.2,
                                                        random_state=2020)
    return x_train, x_test, y_train, y_test


def predict(x_train, x_test, y_train):
    trainWeights = stocGradAscent1(array(x_train), y_train, 500)
    predict = []
    for i in array(x_test):
        predict.append(classifyVector(i, trainWeights))
    predict = array(predict)
    return predict
    # accuracy = mean(predict == y_test)
    # print("accuracy:", accuracy)


def loadCSVfile2():
    tmp = loadtxt("balance-scale.csv", dtype=str, delimiter=",")
    for line in tmp:
        if line[0] == 'L':
            line[0] = 0
        else:
            line[0] = 1
    data = tmp[1:, 1:].astype(float)  # 加载数据部分
    label = tmp[1:, 0].astype(float)  # 加载类别标签部分
    # print(tmp)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    return x_train, x_test, y_train, y_test  # 返回array类型的数据


# dataArr, LabelMat = loadDataSet()
# weights = stocGradAscent1(array(dataArr), LabelMat)
# plotBestFit(weights)

# multiTest()
# x_train, x_test, y_train, y_test = loadIris()
sum_accu = 0
for i in range(10):
    x_train, x_test, y_train, y_test = loadCSVfile2()
    x_train = x_train.tolist()
    x_test = x_test.tolist()
    for inX in x_train:
        inX.append(1)
    for inX in x_test:
        inX.append(1)
    x_train = array(x_train)
    x_test = array(x_test)
    pred = predict(x_train, x_test, y_train)
    accuracy = mean(pred == y_test)
    sum_accu += accuracy
    print("%d accuracy: %f" % (i, accuracy))
print("mean accuracy: %f" % (sum_accu / 10))
