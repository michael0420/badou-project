# read dataset
from tabnanny import verbose
from tensorflow.keras.datasets import mnist

(traindata, trainlabel), (testdata, testlabel) = mnist.load_data()


# build model
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()

network.add(layers.Dense(512, actvation = 'relu', input_shape = (28,28)))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# reshape input
traindata = traindata.reshape((60000, 28*28))
traindata = traindata.astype('float32') / 255

testdata = testdata.reshape((10000, 28*28))
testdata = testdata.astype('float32') / 255

# change labels to one-hot
from tensorflow.keras.utils import to_categorical

trainlabel = to_categorical(trainlabel)
testlabel = to_categorical(testlabel)

# train model
network.fit(traindata, trainlabel, epochs = 5, batch_size = 128)

# check testdata according to trained model

# model.evaluate 评估模型，不输出预测结果
test_loss, test_acc = network.evaluate(testdata, testlabel, verbose=1)

# model.predict 输出预测结果
# res = network.predict(testdata, batch_size = 1)