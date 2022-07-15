import Vgg16Net
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import cv2

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
images = test_images[5000:5001] / 255
images = tf.map_fn(lambda x: cv2.resize(x.numpy(), (224, 224)), images)
label = test_labels[5000]

model = Vgg16Net.vgg16_net(10)
model.load_weights("model/vgg16_model_weights")
predict_model = Sequential([model, Softmax()])
result = np.argmax(predict_model.predict(images)[0])
print(f"True label is {label}, predict label is {result}")