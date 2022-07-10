import numpy as np
from sklearn.metrics import log_loss
from tensorflow.keras.datasets import mnist
import cv2

class MyLayer:
    
    def __init__(self, node_number, activate_fn) -> None:
        if type(node_number) == tuple:
            self.node_number = node_number[0] * node_number[1]
        else:
            self.node_number = node_number
        
        if activate_fn is not None:
            self.activate_fn = activate_fn
        else:
            self.activate_fn = lambda x:x
    def __str__(self) -> str:
        return "MyLayer: {}, {}".format(self.node_number, self.activate_fn.__name__)

class MyNetwork:

    def __init__(self, input_layer: MyLayer, hidden_layer: MyLayer, out_layer: MyLayer, epochs, learn_rate=0.5) -> None:

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer
        self.epochs = epochs
        self.learn_rate = learn_rate

        self.loss_fn = MyNetwork.cross_entropy

        wih_h = hidden_layer.node_number
        wih_w = input_layer.node_number
        self.wih = np.random.random((wih_h, wih_w)) - 0.4999

        who_h = out_layer.node_number
        who_w = hidden_layer.node_number
        self.who = np.random.random((who_h, who_w)) - 0.4999

    def __str__(self) -> str:

        return "MyNetwork: i:{input_layer}, h:{hidden_layer}, o:{out_layer}, epochs:{epochs}, learn_rate:{learn_rate}" \
               "loss:{loss_fn}, wih: {wih}, who: {who}".format(
                input_layer=self.input_layer,
                hidden_layer=self.hidden_layer,
                out_layer=self.out_layer,
                epochs=self.epochs,
                loss_fn=self.loss_fn.__name__,
                wih=self.wih,
                who=self.who,
                learn_rate=self.learn_rate,)

    '''
    训练
    '''
    def train(self, x_train, y_train):

        # print(x_train.shape)
        # print(y_train.shape)

        
        for i in range(self.epochs):
            loss_value = 0
            for j in range(x_train.shape[0]):
                ### input -> hidden
                inputs = np.reshape(x_train[j], (x_train[j].shape[0]*x_train[j].shape[0], ))
                h_ins = self.wih @ inputs
                h_outs = self.hidden_layer.activate_fn(h_ins)

                ### hidden -> output
                o_ins = self.who @ h_outs
                o_outs = self.out_layer.activate_fn(o_ins)
                # print(o_outs)

                ### loss 
                y_train_labels = np.zeros((y_train.size, self.out_layer.node_number), dtype=y_train.dtype)
                for k in range(y_train_labels.shape[0]):
                    y_train_labels[k, y_train[j]] = 1

                loss_value += self.loss_fn(y_train_labels[j], o_outs)
                # print(j, " : ", loss_value)

                o_errors = y_train_labels[j] - o_outs
                h_errors = np.dot(self.who.T, o_errors * o_outs * (1 - o_outs))

                ### update weights
                self.who += self.learn_rate * np.dot(
                        np.reshape((o_errors * o_outs * (1 - o_outs)), (-1, 1)),
                        np.reshape(h_outs, (1, -1))
                    )
                self.wih += self.learn_rate * np.dot(
                        np.reshape((h_errors * h_outs * (1 - h_outs)), (-1, 1)),
                        np.reshape(inputs, (1, -1))
                    )
            print(f"epoch[{i}]: loss is {loss_value/x_train.shape[0]}")

    '''
    推理
    '''
    def predict(self, x):
        res = []
        for i in range(x.shape[0]):
                ### input -> hidden
                inputs = np.reshape(x[i], (x[i].shape[0]*x[i].shape[0], ))
                h_ins = self.wih @ inputs
                h_outs = self.hidden_layer.activate_fn(h_ins)

                ### hidden -> output
                o_ins = self.who @ h_outs
                o_outs = self.out_layer.activate_fn(o_ins)
                res.append(np.argmax(o_outs))
        return np.array(res)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        return np.maximum(x, 0)

    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    '''
    均方误差
    '''
    def mse(true_labels, predicted_labels):
        return np.sum((predicted_labels - true_labels)**2) / len(predicted_labels)

    '''
    交叉熵
    '''
    def cross_entropy(true_labels, predicted_labels):
        return log_loss(true_labels, predicted_labels)

    def softmax(labels):
        np.exp(labels)
        return labels / np.sum(labels)


network = MyNetwork(
    MyLayer((28, 28), None),
    MyLayer(512, MyNetwork.sigmoid),
    MyLayer(10, MyNetwork.sigmoid),
    5
)
# print(network)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
network.train(x_train[0:1000], y_train[0:1000])

image = x_test[0:1]
image_number = y_test[0:1]
cv2.imshow("predict image", image[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
print("predict number is : ", network.predict(image))
print("true number is : ", image_number)