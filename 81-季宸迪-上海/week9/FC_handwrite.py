# Handwrite of Full Connection Neural Network (one hidden layer)

import numpy as np
from scipy import special

class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, learningrate):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
    
        self.lr = learningrate

        # wih
        self.w1 = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # who
        self.w2 = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # biasih
        self.b1 = np.random.rand(self.hnodes, 1) - 0.5
        # biasho
        self.b2 = np.random.rand(self.onodes, 1) - 0.5
        
        # expit: sigmoid
        self.activation_function = lambda x: special.expit(x)

    def forward_propagation(self, inputs_list):
        # inputs x
        self.a1 = np.array(inputs_list, ndmin=2).T # (self.inodes * 1)
        # hidden_inputs
        self.z2 = np.dot(self.w1, self.a1) + self.b1 # (self.hnodes * 1)
        # hidden_outpust
        self.a2 = self.activation_function(self.z2) # (self.hnodes * 1)
        # final_inputs
        self.z3 = np.dot(self.w2, self.a2) + self.b2 # (self.onodes * 1)
        # final_outputs
        self.a3 = self.activation_function(self.z3)

    def backward_propagation(self, targets_list):
        # targets y
        self.targets = np.array(targets_list, ndmin=2).T # (self.onodes * 1)
        # final_outputs_error
        self.da3 = self.targets - self.a3 # MSE Loss
        # final_inputs_error = final_outputs_error * final_outputs * (1 - final_outputs)
        self.dz3 = self.da3 * self.a3 * (1 - self.a3)
        # biasho_error
        self.db2 = self.dz3
        # who_error
        self.dw2 = np.dot(self.dz3, np.transpose(self.a2))
        # hidden_outputs_error
        self.da2 = np.dot(self.w2.T, self.dz3)
        # hidden_inputs_error
        self.dz2 = self.da2 * self.a2 * (1 - self.a2)
        # biasih_error
        self.db1 = self.dz2
        # wih_error
        self.dw1 = np.dot(self.dz2, np.transpose(self.a1))
        # Gradient Descent
        self.w2 += self.lr * self.dw2
        self.b2 += self.lr * self.db2
        self.w1 += self.lr * self.dw1
        self.b1 += self.lr * self.db1

    def train(self, inputs, targets):
        self.forward_propagation(inputs)
        self.backward_propagation(targets)

    def predict(self, inputs):
        self.forward_propagation(inputs)
        return self.a3

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open(r'D:\Python\AI\Homework\week9\NeuralNetWork_0\mnist_train.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        # all_values的第一个值是该图片的分类数字，故去除
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01/output_nodes
        targets[int(all_values[0])] = 1 - (0.01/output_nodes) * (output_nodes-1)
        n.train(inputs, targets)

test_data_file = open(r'D:\Python\AI\Homework\week9\NeuralNetWork_0\mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.predict(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)

