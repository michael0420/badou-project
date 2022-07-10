from tabnanny import verbose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(type(x_train))
print(type(y_train))
print(y_train.shape)

# print(x_train.shape)
# print(y_train.shape)
# print(np.unique(y_train))
# print(x_test[0].shape)
# print(y_test.shape)
cv2.imshow("image 0", x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

x_train, x_test = x_train / 255, x_test / 255

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

# print(x_test.shape)

classes = model.predict(x_test[0:1,:,:])
print(classes)
print(np.argmax(classes))