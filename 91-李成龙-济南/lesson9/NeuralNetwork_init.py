import scipy.special  # 导入特殊函数
import numpy as np

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化神经网络，设置输入层，隐藏层，输出层的节点个数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 初始化权重矩阵
        """
        wih 表示输入层和隐含层节点间链路权重矩阵 
        who 表示隐含层和输出层节点间链路权重矩阵
        """
        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 定义激活函数
        """
        scipy.special.expit表示sigmoid函数
        """
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self):
        # 更新节点链路权重
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

    """
    输入数据测试一下网络，当然程序运行的结果并没有什么实际意义。
    """

input_nodes = 3
hidde_nodes = 3
output_nodes = 3

learning_rate = 0.3

net = NeuralNetwork(input_nodes, hidde_nodes, output_nodes, learning_rate)

net.query([1, 3, 5])
        




