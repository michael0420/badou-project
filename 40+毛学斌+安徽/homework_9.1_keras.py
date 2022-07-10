from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# import cv2
import numpy as np


# 1导入数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape:', train_images.shape)
n1, h1, w1 = train_images.shape
print('train_labels.shape:', train_labels.shape)
print('test_images.shape', test_images.shape)
n2, h2, w2 = test_images.shape
print('test_labels.shape', test_labels.shape)

# 2数据集归一化处理，标签改为one hot 编码
train_images = train_images.reshape(n1, h1 * w1)
train_images = train_images.astype('float32') / 255  # 归一化，转化为0-1之间
test_images = test_images.reshape(n2, h2 * w2)
test_images = test_images / 255

print('before to_categorical:', train_labels[0])
train_labels = to_categorical(train_labels)  # 从1个具体值转化为1组类别标签的形式
print('after to_categorical:', train_labels[0])
test_labels = to_categorical(test_labels)

# 3搭建网络
network = models.Sequential()  # 创建一个串联的空白网络
network.add(layers.Dense(512, activation='relu', input_shape=(h1*w1,)))  # 添加全连接层
network.add(layers.Dense(10, activation='softmax'))  # 添加输出层
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])  # 定义优化器、损失函数、评价指标
# 4训练  epochs循环几次， batch_size一次训练几个
network.fit(train_images, train_labels, epochs=1, batch_size=1000)

# 5验证 verbose=1打印日志
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss:', test_loss)  # 损失
print('test_acc:', test_acc)  # 正确率

# 6预测（推理）
digit = test_images[1]
print(digit.shape)
img = np.uint8(digit * 255).reshape(28, 28)
plt.imshow(img, cmap=plt.cm.binary)  # 二值图
plt.show()
res = network.predict(digit.reshape(1, 28*28))
num = np.argmax(res)
print('the number for picture is :', num)


# 方法二
# img = cv2.imread('my_own_6.png', 0)
# cv2.imshow('', img)
# cv2.waitKey(0)
# print('image.shape:', img.shape)
# img_flat = img.reshape(1, 28*28) / 255
# # print(img_flat.shape)
# res = network.predict(img_flat)  # 预测
# print('预测数据为：\n', res)
# num = np.argmax(res)
# print('the number is {}'.format(num))
