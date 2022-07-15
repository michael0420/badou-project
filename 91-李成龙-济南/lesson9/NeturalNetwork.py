"""
task: 从零完整实现手写数字识别神经网络搭建
Author：91-李成龙-济南
Data： 2022-07-02
"""
import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        # pow (x,y)函数返回的是x的y次方的值。
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, input_list, target_list):
        # 把input_list， target_list转换为numpy支持的二维矩阵， T表示转置操作。

        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏层的神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号经过隐藏层后产生的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的神经元对隐藏层输入的信号做激活函数后得到的输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算损失函数  （这里简单计算其误差）
        outputs_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, outputs_errors * final_outputs * (1 - final_outputs))
        # 根据误差更新权重
        self.who += self.lr * np.dot(outputs_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        pass

    def query(self, input):
        # 根据输入数据，计算结果

        # 计算隐藏层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层从隐藏层接收到的信号量

        output_inputs = np.dot(self.who, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)
        print(output_outputs)
        return output_outputs


# 初始化网络

input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
train_data_file = open("dataset/mnist_train.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()
epochs = 10
for e in range(epochs):
    # 把数据依靠','区分开，并依次读入
    for record in train_data_list:
        all_values =record.split(',')
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets=np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

"""
最后把所有测试图片都输入网络，看看检测效果
"""
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为：", correct_number)
    # 预处理数字图片，归一化
    inputs = (np.asfarray(all_values[1:])) /255.0 *0.99 + 0.01
    # 让训练好的网络判断图片对应的数字，推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("the output  result is:", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
# 计算图片判断的成功率
scores_array = np.asarray(scores)  # 将列表转化为数组
print("performance: = ", scores_array.sum()/scores_array.size)   #numpy.size(a, axis=None)  array.size不指定参数是返回数组元素个数

