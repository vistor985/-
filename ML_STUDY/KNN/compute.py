import numpy as np
import sklearn.datasets as datasets  # 数据集模块
from sklearn.model_selection import train_test_split  # 划分训练集和验证集
import time

# 读取数据集
from ML_STUDY.KNN.knn import KNN  # 看个人的项目路径

X, y = datasets.load_digits(return_X_y=True)
# 随机划分训练集和验证集,使用sklearn中的方法
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_test.shape)
# KNN最近邻进行分类(k默认为3)
knn = KNN(X_test, X_train, y_train)  # 后面依次缺省了,k=3,p=2(欧式距离)
# 获取程序运行时间
start = time.process_time()  # 打开文件的名称Python 3.8 已移除 clock() 方法 可以使用 time.perf_counter() 或 time.process_time() 方法替
pred = knn.kNN()
end = time.process_time()
# 打印程序运行时间
print('Running time: %f Seconds' % (end - start))
# 分类准确率
accuracy = np.mean(pred == y_test)
print('准确率:', accuracy)
