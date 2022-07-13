'''keras训练过程'''

import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import keras.datasets.mnist as mnist

# 1）加载数据集
(train_image, train_label), (test_image, test_label) = mnist.load_data()

# 2）验证数据集的性质
print(train_image.shape)
print(train_label.shape)
print(test_image.shape)
print(test_label.shape)
plt.imshow(train_image[0])

# 3）初始化一个模型
model = keras.Sequential()
model.add(layers.Flatten())  # (60000, 28, 28)  ----> (600000, 28*28)
# 建立全链接层， 使用relu激活
model.add(layers.Dense(64, activation='relu'))
# 添加一个分类层，使用softmax激活。输出0-9是个数字，所以单元数为10
model.add(layers.Dense(10, activation='softmax'))

# 4）编译模型
# 当label是顺序编码的时候，计算交叉熵是 sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
              )

# 5）训练模型
model.fit(train_image, train_label, epochs=50, batch_size=512)
# batch_size = 512, 一个batch一个batch的去训练，不是将所有数据拿进去训练
# 原因：计算机的性能或者说计算机的内存容量在处理大型数据的时候，比如说图片数据的时候，
# 将全部数据加载进去，可能会引起内存爆炸。

model.evaluate(train_image, train_label)
model.evaluate(test_image, test_label)

# 预测test数据集的前10张图片
model.predict(test_image[:10])
# 预测的
np.argmax(model.predict(test_image[:10]), axis=1)
# 实际的
test_label[:10]

# 模型的优化   -----  进行过拟合
model = keras.Sequential()
model.add(layers.Flatten())  # (60000, 28, 28)  ----> (600000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
              )

model.fit(train_image, train_label, epochs=50, batch_size=512, validation_data=(test_image, test_label))

# 模型的再优化   ---- 增加过拟合
model = keras.Sequential()
model.add(layers.Flatten())  # (60000, 28, 28)  ----> (600000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
              )

model.fit(train_image, train_label, epochs=200, batch_size=512, validation_data=(test_image, test_label))


# 模型预测
# 预测test数据集的前10张图片
model.predict(test_image[:10])
# 预测的
np.argmax(model.predict(test_image[:10]), axis=1)
# 实际的
print(test_label[:10])

