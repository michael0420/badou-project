# import warnings  # 消除一部分可以执行的警告
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import \
    Input, DepthwiseConv2D, Conv2D, Activation, Dropout, \
    Reshape, BatchNormalization, GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def depthwiseConvBlock(inputs, filters, depth_multiplier=1,
                       strides=(1, 1), block_id=1):
    """深度可分离卷积模块，输入/通道数/深度分离后的层数/步长/命名代号"""
    # 1、DW逐个通道分离后 卷积 depthwise Conv
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same',
                        depth_multiplier=depth_multiplier, use_bias=False,
                        name='dw_conv_%d' % block_id)(inputs)
    x = BatchNormalization(name='dw_conv_%d_BN' % block_id)(x)
    x = Activation(relu6, name='de_conv_%d_relu' % block_id)(x)
    # 2、PW逐个点卷积,pointwise Conv
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               strides=(1, 1), name='pw_conv_%d' % block_id)(x)
    x = BatchNormalization(name='pw_conv_%d_BN' % block_id)(x)
    x = Activation(relu6, name='pw_conv_%d_relu' % block_id)(x)
    return x


def convBlock(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """正常卷积模块，卷积+标准化+激活"""
    x = Conv2D(filters, kernel, strides=strides, padding='same',
               use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_BN')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNet(input_shape=(224, 224, 3),
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):
    """ 输出MobileNet模型"""
    img_input = Input(shape=input_shape)
    # 计算图： 参照ppt，最后一个模块做了优化
    # 1、步长为2的普通卷积1次： 224,224,3 -> 112,112,32
    x = convBlock(img_input, 32, strides=(2, 2))  # 默认步长为1
    # 2、步长为1的dw卷积+pw： 112,112,32 -> 112,112,64
    x = depthwiseConvBlock(x, 64, depth_multiplier, block_id=1)
    # 3、步长为2的dw卷积+pw： 112,112,64 -> 56,56,128
    x = depthwiseConvBlock(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 4、步长为1的dw卷积+pw： 56,56,128 -> 56,56,128
    x = depthwiseConvBlock(x, 128, depth_multiplier, block_id=3)
    # 5、步长为2的dw卷积+pw： 56,56,128 -> 28,28,256
    x = depthwiseConvBlock(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 6、步长为1的dw卷积+pw： 28,28,256 -> 28,28,256
    x = depthwiseConvBlock(x, 256, depth_multiplier, block_id=5)
    # 7、步长为2的dw卷积+pw： 28,28,256 -> 14,14,512
    x = depthwiseConvBlock(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    # 8、步长为1的dw卷积+pw，循环5次： 14,14,512 -> 14,14,512
    for i in range(5):
        x = depthwiseConvBlock(x, 512, depth_multiplier, block_id=i+7)
    # 9、步长为2的dw卷积+pw + 步长为1的dw卷积+pw： ： 14,14,512 -> 7,7,1024
    x = depthwiseConvBlock(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwiseConvBlock(x, 1024, depth_multiplier, block_id=13)
    # 10、均值池化+FC+softmax激活输出： 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)  # 将四维(n,7,7,1024)压扁成二维(n,1024)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)  # 生维到四维(n,1,1,1024),前还有个数的维度
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_fc')(x)  # 用1*1卷积替代fc
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)  # 变成1维数组便于后面处理

    inputs = img_input
    _model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    return _model


def preprocessInput(x):
    """标准化到-1到+1之间"""
    x /= 255.
    x -= 0.5
    x *= 2
    return x


if __name__ == '__main__':
    # 1、模型实例化并导入参数
    model = MobileNet(input_shape=(224, 224, 3))
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)
    # 2、读入数据并预处理
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)  # 转化成4维
    img_data = preprocessInput(img_data)  # 标准化到-1~+1
    print('Input image shape:', img_data.shape)
    # 3、用模型进行预测
    preds = model.predict(img_data)
    print('预测的类别序号为：', np.argmax(preds))
    print('预测结果为：', decode_predictions(preds, 1))  # 1为TOP1
