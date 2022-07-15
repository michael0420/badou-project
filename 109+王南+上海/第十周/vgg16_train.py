import Vgg16Net
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2
from tensorflow.keras.utils import to_categorical

train_num = 100
test_num = 10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images[:train_num] / 255, test_images[:test_num] / 255
train_labels, test_labels = to_categorical(train_labels)[:train_num], to_categorical(test_labels)[:test_num]
print(train_images.shape)
print(train_images[0].shape)
print(type(train_images[0]))


train_images = tf.map_fn(lambda x: cv2.resize(x.numpy(), (224, 224)), train_images)
test_images = tf.map_fn(lambda x: cv2.resize(x.numpy(), (224, 224)), test_images)


model = Vgg16Net.vgg16_net(len(train_labels[0]))
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(),
    metrics=["accuracy"])
model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          batch_size=20, epochs=10,
          verbose=1)
model.save_weights("model/vgg16_model_weights")
print("end.")


