import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate ):
        self.inode = inputnodes
        self.hnode = hiddennodes
        self.onode = outputnodes

        #设置学习率
        self.lr = learningrate

        self.wih = np.random.rand(self.hnode, self.inode) - 0.5
        self.who = np.random.rand(self.onode, self.hnode) - 0.5
        # self.wih = np.random.normal(0, pow(self.hnode,-0.5), (self.hnode,self.inode))
        # self.who = np.random.normal(0, pow(self.onode,-0.5), (self.onode,self.hnode))

        self.avtivation_funtion = lambda x: scipy.special.expit(x)


        pass

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin = 2).T
        target = np.array(target_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.avtivation_funtion(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.avtivation_funtion(final_inputs)

        #计算误差
        output_errors = target - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 -
                                                                            final_outputs))
        self.who += self.lr * np.dot((output_errors * final_outputs* (1 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors *hidden_outputs * (1-hidden_outputs)),np.transpose(inputs))


        pass

    def test(self, input):
        hidden_inputs = np.dot(self.wih, input)
        hidden_outputs = self.avtivation_funtion(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.avtivation_funtion(final_inputs)
        print(final_outputs)
        return final_outputs


#初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


#读入训练数据
train_data_file = open('mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

#加入epoch,设定网络的训练循环次数
epoch = 5
for e in range(epoch):
    for record in train_data_list:
        all_value = record.split(',')
        inputs = (np.asfarray(all_value[1:])) / 255.0 * 0.99 + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_value[0])] = 0.99
        n.train(inputs,targets)

test_data_file = open('mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []

for record in test_data_list:
    all_value = record.split(',')
    correct_numbers = int(all_value[0])
    print("该图片对应的数字为:",correct_numbers)
    inputs = (np.asfarray(all_value[1:])) / 255 * 0.99 + 0.01
    outputs = n.test(inputs)
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_numbers:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
#判断图片判断的成功率
scores_array = np.asfarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
