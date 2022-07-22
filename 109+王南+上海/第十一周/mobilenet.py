from keras.layers import Conv2D, DepthwiseConv2D, Dense, Input, Flatten
from keras.layers import BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout, Softmax, Reshape
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
import numpy as np


def dw_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = DepthwiseConv2D(kernel_size, strides, padding="same", depth_multiplier=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = relu6(x)
    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = relu6(x)
    return x


def conv_block(x, filters, kernel_size=(1, 1), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = relu6(x)
    return x


def relu6(x):
    return ReLU(max_value=6)(x)


def mobile_net(input_shape=(224, 224, 3), classes=1000):
    input_image = Input(input_shape)
    x = conv_block(input_image, filters=32, kernel_size=(3, 3), strides=(2, 2))
    x = dw_block(x, filters=64)
    x = dw_block(x, filters=128, strides=(2, 2))
    x = dw_block(x, filters=128)
    x = dw_block(x, filters=256, strides=(2, 2))
    x = dw_block(x, filters=256)
    x = dw_block(x, filters=512, strides=(2, 2))
    for _ in range(5):
        x = dw_block(x, filters=512)
    x = dw_block(x, filters=1024, strides=(2, 2))
    x = dw_block(x, filters=1024, strides=(2, 2))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024))(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Softmax()(x)
    x = Reshape((classes,))(x)

    return Model(input_image, x)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = mobile_net()
    model.load_weights('mobilenet_1_0_224_tf.h5')

    img = image.load_img('elephant.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    print(decode_predictions(predictions, 1))