from tensorflow.keras.datasets import mnist
(train_image, train_label), (test_image, test_label) = mnist.load_data()
print('train_image.shape:', train_image.shape)
print('train_label.shape:', train_label.shape)
print('test_image.shape:', test_image.shape)
print('test_label.shape:', test_label.shape)

import matplotlib.pyplot as plt
digital = test_image[0]
plt.imshow(digital, cmap= plt.cm.binary)
plt.show()


from tensorflow.keras import layers
from tensorflow.keras import models


networks = models.Sequential()
networks.add(layers.Dense(512, activation= 'relu', input_shape=(28*28, ) ))
networks.add(layers.Dense(10, activation= 'softmax'))
networks.compile(optimizer='rmsprop', loss= 'categorical_crossentropy',
                 metrics= ['accuracy'])

train_image = train_image.reshape((60000, 28*28))
train_image = train_image.astype('float32') / 255

test_image = test_image.reshape((10000, 28*28))
test_label_image = test_image.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

networks.fit(train_image,train_label,epochs=5,batch_size=128,verbose=1)

test_loss,test_acc = networks.evaluate(test_image, test_label , verbose= 1 )
print('test_loss:', test_loss)
print('test_acc', test_acc)

(train_image, train_label), (test_image, test_label) = mnist.load_data()
digit = test_image[2]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_image= test_image.reshape((10000,28*28))
res = networks.predict(test_image)
for i in range(res[2].shape[0]):
    if res[2][i]==1:
        print("the number for the picture is : ", i)
        break