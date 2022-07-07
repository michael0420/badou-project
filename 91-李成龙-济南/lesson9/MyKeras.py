"""
Task: keras实现手写数字识别
Author： 91-李成龙-济南
Data： 2022.07.01. 01：40
"""
[1]  # 加载数据集

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # 加载mnist数据集

print("train_images.shape：", train_images.shape)  # 打印训练集图像的的大小，训练集的标签
print("train_labels:", train_labels)
print("test_images.shape:", test_images.shape)    # 打印测试集图像的大小，测试集的标签
print("test_labels:", test_labels)

[2]  # 打印一下数据集的某一张图像，检查数据集是否正确。

test_images0 = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(test_images0, cmap=plt.cm.binary)  # matplotlib.cm是 matplotlib库中内置的色彩映射函数。
                                              # matplotlib.cm.[色彩]('[数据集]')即对[数据集]应用[色彩]
plt.show()

[3]  # 搭建网络模型

from tensorflow.keras import models
from tensorflow.keras import layers
#
# net = models.Sequential()   # Sequential()方法是一个容器，描述了神经网络的网络结构，
#                             # 在Sequential()的输入参数中描述从输入层到输出层的网络结构
# net.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
# net.add(layers.Dense(10, activation="softmax"))  # 不用再指定输入的尺寸，因为在创建上一层网络时已经指定过了。
# 也可以向Sequential模型传递一个包含层参数的list，直接构造模型：
net = models.Sequential([layers.Dense(512, activation='relu', input_shape=(28 * 28,)),  # 输入input_shape 用元组表示的话，一个元素后边也得加个逗号
                      layers.Dense(10, activation='softmax')])
net.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

[4]  # 数据预处理 （归一化）
"""
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28(784)个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
"""

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

""""
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot （独热编码）
"""

from tensorflow.keras.utils import to_categorical  # 简单来说，to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。
                                                   # 其表现为将原有的类别向量转换为独热编码的形式。

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:", test_labels[0])
"""
简单来说，to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。
其表现为将原有的类别向量转换为独热编码的形式。
"""

[5]  # 训练神经网络


net.fit(train_images, train_labels, epochs=5, batch_size=128)


[6]  # 测试数据
"""
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
"""
test_loss, test_acc = net.evaluate(test_images, test_labels, verbose=1)  # verbose = 1 不显示日志
print("test_loss:", test_loss)
print("test_acc:", test_acc)

[7]  # 模拟输入一张livedata数据，判断一下网络识别效果

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_image10 = test_images[10]
import matplotlib.pyplot as plt
plt.imshow(test_image10, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32")/255
res = net.predict(test_images)
# print(res)
print(res.shape)
print(res.shape[0])
# print(res[10])
print(res[10].shape[0])  # 应该将shape[0]理解为第一维，shape[1]理解为第二维，同理还有shape[2]、shape[3]等等
for i in range(res[10].shape[0]):
    if (res[10][i]) == 1:
        print("the number of the picture is :", i)
        break
















