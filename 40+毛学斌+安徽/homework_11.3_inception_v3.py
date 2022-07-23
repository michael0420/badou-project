import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, \
    Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


def conv2d_bn(x, filters, row, col, strides=(1, 1), padding='same', name=None):
    """卷积模块，包括卷积+批标准化+relu激活"""
    if name is None:  # 命名用
        conv_name = None
        bn_name = None
    else:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    x = Conv2D(filters, (row, col), strides=strides, padding=padding,
               use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(img_shape=(299, 299, 3), classes=1000):
    """返回inception V3的模型"""
    # 输入299*299*3
    img_input = Input(shape=img_shape)
    # 卷积，大小（3，3），步长2，输出149*149*32
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    # 卷积，大小（3，3），步长1，输出147*147*32
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    # 卷积，大小（3，3），步长1，加padding的卷积，输出147*147*64
    x = conv2d_bn(x, 64, 3, 3)  # 默认padding=same
    # 池化，大小（3，3），步长2，输出73*73*64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 卷积，大小（1，1），步长1，输出73*73*80
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    # 卷积，大小（3，3），步长1，输出71*71*192
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    # 池化，大小（3，3），步长2，输出35*35*192
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 模块1，重复3次，大小35*35，通道192-256-288-288
    # 模块1每一次并联concat（1*1，pool，2个3*3，5*5）
    # 第一步：35x35x 192 -> 35x35x 256
    # 注意：导入别人的参数必须采用相同的顺序结构
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    # 并联通道合并：64+32+96+64=256
    # 因为导入别人训练的参数，所以concatenate必须采用相同的顺序，否则会报错shape不同
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool],
                           axis=3, name='mixed0')
    # 第二步：35x35x 256 -> 35x35x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 仅这一步卷积核改为64了
    # 并联通道合并：64+64+96+64=288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool],
                           axis=3, name='mixed1')  # 注意顺序不能变
    # 第三步：35x35x 288 -> 35x35x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 仅这一步卷积核改为64了
    # 并联通道合并：64+64+96+64=288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pool],
                           axis=3, name='mixed2')

    # 模块2，共5步，大小17x17
    # 其中第一步调整大小，2-5步类似，仅中间通道数不同，第3，4步完全相同
    #  第一步：35x35x 288 -> 17x17x 768
    # 并联concat（3x3，串联2个3x3，最大池化）最后一步步长2调整尺寸
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch3x3db = conv2d_bn(x, 64, 1, 1)
    branch3x3db = conv2d_bn(branch3x3db, 96, 3, 3)
    branch3x3db = conv2d_bn(branch3x3db, 96, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 并联通道合并：384+96+288=768
    x = layers.concatenate([branch3x3, branch3x3db, branch_pool],
                           axis=3, name='mixed3')
    #  第二步：17x17x 768 -> 17x17x 768
    # 并联concat（1x1，7x7，串联2个7x7，均值池化）
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch7x7db = conv2d_bn(x, 128, 1, 1)
    branch7x7db = conv2d_bn(branch7x7db, 128, 7, 1)
    branch7x7db = conv2d_bn(branch7x7db, 128, 1, 7)
    branch7x7db = conv2d_bn(branch7x7db, 128, 7, 1)
    branch7x7db = conv2d_bn(branch7x7db, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    # 并联通道合并：192+192+192+192=768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool],
                           axis=3, name='mixed4')
    #  第三步和第四步相同：17x17x 768 -> 17x17x 768，仅中间通道与第二步不同
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(x, 160, 1, 1)  # 中间通道从128换成160了
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 最终输出通道没变
        branch7x7db = conv2d_bn(x, 160, 1, 1)
        branch7x7db = conv2d_bn(branch7x7db, 160, 7, 1)
        branch7x7db = conv2d_bn(branch7x7db, 160, 1, 7)
        branch7x7db = conv2d_bn(branch7x7db, 160, 7, 1)
        branch7x7db = conv2d_bn(branch7x7db, 192, 1, 7)
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        # 并联通道合并：192+192+192+192=768
        x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool],
                               axis=3, name='mixed'+str(5 + i))
    #  第五步与 2，3，4步类似：17x17x 768 -> 17x17x 768，仅中间通道数不同
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, 192, 1, 1)  # 中间通道从128换成192了
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 最终输出通道没变
    branch7x7db = conv2d_bn(x, 192, 1, 1)
    branch7x7db = conv2d_bn(branch7x7db, 192, 7, 1)
    branch7x7db = conv2d_bn(branch7x7db, 192, 1, 7)
    branch7x7db = conv2d_bn(branch7x7db, 192, 7, 1)
    branch7x7db = conv2d_bn(branch7x7db, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    # 并联通道合并：192+192+192+192=768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db, branch_pool],
                           axis=3, name='mixed7')

    # 模块3，共三步，大小8*8
    # 其中第一步调整大小，2-3步完全相同
    #  第一步：17x17x 768 -> 8x8x 1280
    # 并联concat（3x3，串联7x7+3x3，最大池化）最后一步步长2调整尺寸
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')
    branch7x7a3x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7a3x3 = conv2d_bn(branch7x7a3x3, 192, 1, 7)
    branch7x7a3x3 = conv2d_bn(branch7x7a3x3, 192, 7, 1)
    branch7x7a3x3 = conv2d_bn(branch7x7a3x3, 192, 3, 3,
                              strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 并联通道合并：320+192+768=1280
    x = layers.concatenate([branch3x3, branch7x7a3x3, branch_pool],
                           axis=3, name='mixed8')
    #  第二步，第三步相同：8x8x 1280 -> 8x8x 2048
    # 并联concat（1x1，并联1x3和3x1，先3x3再并联1x3和3x1，平均池化）最后一步步长2调整尺寸
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)  # 这里用到并联
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                       axis=3, name='mixed9_'+str(i))

        branch3x3db = conv2d_bn(x, 448, 1, 1)
        branch3x3db = conv2d_bn(branch3x3db, 384, 3, 3)
        branch3x3db_1 = conv2d_bn(branch3x3db, 384, 1, 3)
        branch3x3db_2 = conv2d_bn(branch3x3db, 384, 3, 1)
        branch3x3db = layers.concatenate([branch3x3db_1, branch3x3db_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 并联通道合并：320 + 384*2 + 384*2 + 192 = 2048
        x = layers.concatenate([branch1x1, branch3x3, branch3x3db, branch_pool],
                               axis=3, name='mixed'+str(9+i))

    # 全局平均池化（降维）后全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    inputs = img_input
    _model = Model(inputs, x, name='inception_v3')
    return _model


if __name__ == '__main__':
    # 1、模型实例化并导入参数
    model = InceptionV3()
    # model.summary()  # 显示模型结构
    model_name = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(model_name)
    # 2、读入数据并预处理
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data, mode='tf')
    # keras中的preprocess_input默认归一到0~1之间，增加mode='tf'后可以归一化到-1~+1之间，同时该函数更改原值
    # 3、用模型进行预测
    print('输入图像的形状为：', img_data.shape)
    preds = model.predict(img_data)
    print('预测结果为：', decode_predictions(preds, 1))
