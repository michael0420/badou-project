import numpy as np
from keras.layers import Conv2D, Dense, BatchNormalization, Input
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import ReLU
from keras import layers
from keras import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding="same", name=None):
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    x = ReLU()(x)
    return x


def inception_v3(input_shape=(299,299,3), classes=1000):

    input_image = Input(input_shape)

    x = conv2d_bn(input_image, 32, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    '''
    Block1 35x35
    '''
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1)
    branch5x5 = conv2d_bn(x, 48, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=3,
                           name="mixed0")

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1)
    branch5x5 = conv2d_bn(x, 48, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1)
    branch5x5 = conv2d_bn(x, 48, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')

    '''
    Block2 17x17
    '''
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, strides=(2, 2), padding='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed3')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1)
    branch7x7 = conv2d_bn(x, 128, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))
    branch7x7dbl = conv2d_bn(x, 128, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1)
        branch7x7 = conv2d_bn(x, 160, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, (1, 7))
        branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))
        branch7x7dbl = conv2d_bn(x, 160, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1)
    branch7x7 = conv2d_bn(x, 192, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))
    branch7x7dbl = conv2d_bn(x, 192, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    '''
    Block3 8x8
    '''
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, strides=(2, 2), padding='valid')
    branch7x7x3 = conv2d_bn(x, 192, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3,
        name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1)
        branch3x3 = conv2d_bn(x, 384, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, (1, 3))
        branch3x3_2 = conv2d_bn(branch3x3, 384, (3, 1))
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=3,
            name=f'mixed9_{i}')

        branch3x3dbl = conv2d_bn(x, 448, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, (1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, (3, 1))
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2],
            axis=3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1)

        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name=f'mixed{9+i}')

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    return Model(input_image, x)



def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':

    model = inception_v3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img = image.load_img('elephant.jpg', target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    print(decode_predictions(predictions))