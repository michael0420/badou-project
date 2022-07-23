from keras.models import Sequential
from keras.layers import \
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential(
        Conv2D(filters=96,  # 卷积核个数
               kernel_size=(11, 11),  # 卷积核大小
               strides=(4, 4),  # 步长
               padding='valid',  # padding无
               input_shape=input_shape,
               activation='relu'),  # 激活函数
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2),
                     padding='valid'),
        Conv2D(filters=256,
               kernel_size=(5, 5),
               strides=(1, 1),
               padding='same',
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2),
                     padding='valid'),
        Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               activation='relu'),
        Conv2D(filters=384,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               activation='relu'),
        Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               activation='relu'),
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.25),
        Dense(1024, activation='relu'),
        Dropout(0.25),
        Dense(output_shape, activation='softmax')
    )
    return model
