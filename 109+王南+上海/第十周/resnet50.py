from keras import layers
from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Flatten
from keras.layers import ReLU, BatchNormalization, ZeroPadding2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np


def identity_block(input_tensor, kernel_size, filters):
    f1, f2, f3 = filters

    block = Conv2D(f1, (1, 1))(input_tensor)
    block = BatchNormalization()(block)
    block = ReLU()(block)
    block = Conv2D(f2, kernel_size, padding="same")(block)
    block = BatchNormalization()(block)
    block = ReLU()(block)
    block = Conv2D(f3, (1, 1))(block)
    block = BatchNormalization()(block)

    block = layers.add([block, input_tensor])
    block = ReLU()(block)

    return block


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    f1, f2, f3 = filters

    block1 = Conv2D(f1, (1, 1), strides)(input_tensor)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)
    block1 = Conv2D(f2, kernel_size, padding="same")(block1)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)
    block1 = Conv2D(f3, (1, 1))(block1)
    block1 = BatchNormalization()(block1)

    block2 = Conv2D(f3, (1, 1), strides, padding="same")(input_tensor)
    block2 = BatchNormalization()(block2)

    block = layers.add([block1, block2])
    block = ReLU()(block)

    return block


def resnet50(input_shape=(224, 224, 3), classes=1000):
    input = Input(input_shape)
    m = ZeroPadding2D((3, 3))(input)
    m = Conv2D(64, (7, 7), strides=(2, 2))(m)
    m = BatchNormalization()(m)
    m = ReLU()(m)
    m = MaxPooling2D((3, 3), strides=(2, 2))(m)

    m = conv_block(m, 3, [64, 64, 256], strides=(1, 1))
    m = identity_block(m, 3, [64, 64, 256])
    m = identity_block(m, 3, [64, 64, 256])

    m = conv_block(m, 3, [128, 128, 512])
    m = identity_block(m, 3, [128, 128, 512])
    m = identity_block(m, 3, [128, 128, 512])
    m = identity_block(m, 3, [128, 128, 512])

    m = conv_block(m, 3, [256, 256, 1024])
    m = identity_block(m, 3, [256, 256, 1024])
    m = identity_block(m, 3, [256, 256, 1024])
    m = identity_block(m, 3, [256, 256, 1024])
    m = identity_block(m, 3, [256, 256, 1024])
    m = identity_block(m, 3, [256, 256, 1024])

    m = conv_block(m, 3, [512, 512, 2048])
    m = identity_block(m, 3, [512, 512, 2048])
    m = identity_block(m, 3, [512, 512, 2048])

    m = AveragePooling2D((7, 7))(m)

    m = Flatten()(m)
    m = Dense(classes, activation='softmax')(m)

    m = Model(input, m)

    return m


if __name__ == "__main__":

    model = resnet50()
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    img = image.load_img("elephant.jpg", target_size=(224, 224))
    inputs = image.img_to_array(img)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = preprocess_input(inputs)
    predictions = model.predict(inputs)

    print(decode_predictions(predictions))