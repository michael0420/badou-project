import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def vgg16_net(out_features):
    p = 8
    p1 = 2**7
    return Sequential([
        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)，再2X2最大池化，输出net为 (112,112,64)。
        Conv2D(64/p, 3, strides=(1, 1), activation="relu", padding="same", input_shape=(224, 224, 3)),
        Conv2D(64/p, 3, strides=(1, 1), activation="relu", padding="same"),
        MaxPool2D(pool_size=2, strides=(2, 2)),
        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出 net为(56,56,128)
        Conv2D(128/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(128/p, 3, strides=(1, 1), activation="relu", padding="same"),
        MaxPool2D(pool_size=2, strides=(2, 2)),
        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net 为(28,28,256)
        Conv2D(256/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(256/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(256/p, 3, strides=(1, 1), activation="relu", padding="same"),
        MaxPool2D(pool_size=2, strides=(2, 2)),
        # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)，再2X2最大池化，输出net 为(14,14,512)
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        MaxPool2D(pool_size=2, strides=(2, 2)),
        # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net 为(7,7,512)
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        Conv2D(512/p, 3, strides=(1, 1), activation="relu", padding="same"),
        MaxPool2D(pool_size=2, strides=(2, 2)),
        ### FC层分类
        # Flatten(),
        # Dense(4096/p1, activation="relu"),
        # Dropout(0.5, trainable=False),
        # Dense(4096/p1, activation="relu"),
        # Dropout(0.5, trainable=False),
        # Dense(out_features),
        ### 卷积替代FC层分类
        Conv2D(4096/p1, 7, strides=(1, 1), activation="relu"),
        Dropout(0.5, trainable=False),
        Conv2D(4096/p1, 1, strides=(1, 1), activation="relu"),
        Dropout(0.5, trainable=False),
        Conv2D(out_features, 1, strides=(1, 1), activation=None),
        Flatten()
    ])
