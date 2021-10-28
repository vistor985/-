import matplotlib.pyplot as plt
import sklearn.datasets as datasets  # 数据集模块

# 读取数据集
X, y = datasets.load_digits(return_X_y=True)

for i in range(32):
    plt.subplot(4, 8, i + 1)
    img = X[i, :].reshape(8, 8)
    plt.imshow(img)
    plt.title(y[i])
    plt.axis("off")
    plt.subplots_adjust(hspace=0.3)  # 微调行间距
plt.show()
