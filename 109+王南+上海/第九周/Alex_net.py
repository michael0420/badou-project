from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import cv2

# 设置训练样本数（总共50000）
train_num = 1000
# 设置测试样本数（总共10000）
test_num = 100

# 取训练和测试样本并归一化
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images[0:train_num] / 255, test_images[0:test_num] / 255
train_labels, test_labels = to_categorical(train_labels)[0:train_num], to_categorical(test_labels)[0:test_num]

# print(type(train_images[0]))
print(train_images.shape)
print(test_images.shape)
# print(type(train_labels[0]))
print(train_labels.shape)
print(test_labels.shape)

# 图片处理为224x224，适配AlexNet输入图片的大小
train_images_input = []
test_images_input = []
for i in range(train_num):
    train_images_input.append(cv2.resize(train_images[i], (224, 224)))
for i in range(test_num):
    test_images_input.append(cv2.resize(test_images[i], (224, 224)))
train_images_input = np.array(train_images_input)
test_images_input = np.array(test_images_input)
print(train_images_input.shape)


# 构建AlexNet模型
model = Sequential()
model.add(Conv2D(filters=96, kernel_size=11, strides=(4, 4), padding="valid",
                 input_shape=(224, 224, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=5, strides=(1, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
model.add(Conv2D(filters=384, kernel_size=3, strides=(1, 1), padding="same", activation="relu"))
model.add(Conv2D(filters=384, kernel_size=3, strides=(1, 1), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

# 训练模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
modelCheckpoint = ModelCheckpoint(
    "data/ep-{epoch:03d}-acc-{acc:.3f}.h5",
    monitor="acc",
    save_best_only=True,
    save_weights_only=True,
    period=1)
reduceLROnPlateau = ReduceLROnPlateau(monitor="acc", factor=0.1, patience=2, epsilon=0.001, cooldown=0, min_lr=0.0001)
# earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=1)
model.fit(train_images_input, train_labels, batch_size=100, epochs=10,
          callbacks=[modelCheckpoint, reduceLROnPlateau])
model.save_weights("data/alex_net_1000.h5")

# 测试模型
model.evaluate(test_images_input, test_labels, verbose=2)

# 预测
print(model.predict(cv2.resize(test_images[11], (224, 224, 3))))

