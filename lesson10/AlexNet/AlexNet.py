from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization


def AlexNet(input_ashape=(224,224,3), output_shape=2,):
    model = Sequential()
    #实际AlexNet论文里实现的是用两个GPU分别训练的，所以这里数据量减半，用单个cpu进行联系。
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(55, 55),
            strides=(4, 4),
            padding="valid",  # without padding
            activation="relu"
        )
    )

    model.add(BatchNormalization)
    # 批标准化  批量归一化
    # 批标准化一般用在非线性映射（激活函数）之前，对y= Wx + b进行规范化，
    # 使结果(输出信号的各个维度)的均值都为0, 方差为1, 让每一层的输入有一个稳定的分布会有利于网络的训练。
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid"
        )
    )

    model.add(Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding= "same",
        activation="relu"
    )
              )
    model.add(BatchNormalization)

    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid"
    )
              )

    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu"
    )
              )

    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu"
    )
              )

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu"
    )
              )

    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid"
    )
              )
    # 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, ))