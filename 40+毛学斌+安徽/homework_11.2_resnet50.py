import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, Dense, Activation, BatchNormalization, Flatten
from keras.applications.imagenet_utils import \
    preprocess_input, decode_predictions


def identity_block(input_tensor, kernel, filters, stage, block):
    """一致模块，输入和输出大小相等"""
    filter1, filter2, filter3 = filters  # 把输入的列表中的3个值分给3个变量
    conv_name_base = 'identity' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'  # 命名用
    # 1、左分支:卷积+BN+Relu 两次后 卷积+BN
    # 共3步，1，3卷积都是（1，1），第2步卷积核外部传入，这三步，步长都是（1，1）
    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 2、右分支shortcut:空 即 输入
    # 3、左右分支并联合并+relu
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel, filters, stage, block, strides=(2, 2)):
    """卷积模块-输入和输出大小不相等"""
    filter1, filter2, filter3 = filters
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 1、左分支:卷积+BN+Relu 两次后 卷积+BN
    # 共3步，1，3卷积都是（1，1），第2步卷积核外部传入，步长：第一步外部传入，默认（2，2），其余都是（1，1)
    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 2、右分支shortcut:卷积+BN
    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    # 3、左右分支并联合并+relu
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=(224, 224, 3), classes=1000):
    """输出ResNet模块"""
    img_input = Input(shape=input_shape)  # 输出（224,224,3）
    # 1、增加0值padding
    x = ZeroPadding2D((3, 3))(img_input)  # 输出（230,230,3）
    # 2、卷积+BN+Relu+MaxPool
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)  # 输出（112,112,64）
    x = BatchNormalization(name='bn_conv1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 输出（55,55,64）
    # 3、一个conv_block模块+两个identity_block模块,本次步长（1，1）  输出（55,55,256）
    x = conv_block(x, (3, 3), [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='b')
    x = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='c')
    # 4、一个conv_block模块+三个identity_block模块，步长默认（2,2）  输出（28,28,512）
    x = conv_block(x, (3, 3), [128, 128, 512], stage=3, block='a')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='b')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='c')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='d')
    # 5、一个conv_block模块+五个identity_block模块，步长默认（2,2）  输出（14,14,1024）
    x = conv_block(x, (3, 3), [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='f')
    # 6、一个conv_block模块+二个identity_block模块，步长默认（2,2）  输出（7,7,2048）
    x = conv_block(x, (3, 3), [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='c')
    # 7、均值池化+flatten+FC+输出
    x = AveragePooling2D((7, 7), name='ave_pool')(x)  # 输出(1,1,2048)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    _model = Model(img_input, x, name='resnet50')
    return _model


if __name__ == '__main__':
    # 1、模型实例化并导入参数
    model = ResNet50()
    # model.summary()  # 展示一下模型结构
    model_name = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(model_name)
    # 2、读入数据并预处理
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)  # 转化为数组
    img_data = np.expand_dims(img_data, axis=0)  # 3维升4维
    img_data = preprocess_input(img_data)  # 标准化处理成-1~+1
    # 3、用模型进行预测
    print('输入图像的形状：', img_data.shape)
    preds = model.predict(img_data)
    print('预测结果为：', decode_predictions(preds, 1))
