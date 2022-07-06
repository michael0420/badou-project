# 用keras实现简单的手写体识别

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import random

# 加载手写数字的数据到内存
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print('train_images.shape = ',train_images.shape)
# print('tran_labels = ', train_labels)
# print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

# 使用tensorflow.Keras搭建一个有效识别图案的神经网络
network = models.Sequential()        # models.Sequential():表示把每一个数据处理层串联起来
# network.add(layers.Conv2D(filters=2, kernel_size=(3*3), strides=2, padding='SAME'))
network.add(layers.Dense(512, activation='relu'))   # 加入正则化方式后效果变差     # layers.Dense全连接层  activation可选 relu 、softmax、 sigmoid、 tanh等
# network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(1e-2)))   # 多加入一层全连接层效果也变差
network.add(layers.Dense(10, activation='softmax'))

# network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))     # layers.Dense全连接层  activation可选 relu 、softmax、 sigmoid、 tanh等
# network.add(layers.Dense(10, activation='softmax'))

# 网络编译   optimizer优化器，也可选SGD   loss损失函数    metrics度量指标
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 在把数据输入到网络模型之前，把数据做归一化处理:
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# to_categorical  用one_hot编译把图片对应的标记也做一个更改，例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0]
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# network.fit 开始网络训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)  # epochs:每次计算的循环是五次  batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算

network.summary() #打印神经网络结构，统计参数数目

# 测试数据输入，检验网络学习后的图片识别效果.
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss = ',test_loss) 
print('test_acc = ', test_acc)


# 输入一张手写数字图片到网络中，看看它的识别效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
aa = []
for i in range(10):
    aa.append(random.randint(0, 1000))
for i in range(10):
    plt.subplot(2, 5, i+1), plt.imshow(test_images[aa[i]], cmap=plt.cm.binary)
    plt.xticks([]), plt.yticks([])
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)   # 对测试集进行预测

result = []
for j in range(10):
    for i in range(res[aa[j]].shape[0]):
        if (res[aa[j]][i] == 1):
            result.append(i)
            break

labels = []
for i in range(10):
    labels.append(test_labels[aa[i]])
print('labels',labels)

plt.suptitle('labels = '+str(labels)+'\n'+'result = '+str(result))
print("the number for the pictures are : ", result)
plt.savefig('result.png')
plt.show()


