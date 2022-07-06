from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt

#控制显存的使用
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


#读数据
(train_imgs,train_labels),(test_imgs,test_labels) = mnist.load_data()

train_imgs = train_imgs.reshape((60000,28*28)).astype('float')/255
test_imgs = test_imgs.reshape((10000,28*28)).astype('float')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#搭建网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#训练网络
network.fit(train_imgs,train_labels,epochs=10,batch_size=128)

#传入测试集，检验正确率
test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

#预测第10张图
(train_imgs,train_labels),(test_imgs,test_labels) = mnist.load_data()
digit = test_imgs[10]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


test_10 = test_imgs[10].reshape((1, 28*28))

y_pre = network.predict(test_10)
print(y_pre,test_labels[10])

