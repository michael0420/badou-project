import numpy
import scipy.special
import matplotlib.pyplot as plt

class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #初始化网络，设置输入层，中间层，和输出层节点数、学习率
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化权重矩阵，我们有两个权重矩阵，
        # 一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        # 一个是who,表示中间层和输出层间链路权重形成的矩阵
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5    # random.rand一组服从“0~1”均匀分布的随机样本值
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))   # pow() 方法返回x的y次方）的值。  random.normal一组正态分布的随机样本值
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # 每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)   # expit（x）= 1 /（1 + exp（-x））(sigmoid函数)
        pass
        
    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        #根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass
        
    def query(self, inputs):
        #根据输入数据计算并输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
        
#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 512
output_nodes = 10
learning_rate = 0.2
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 20
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01   # numpy.asfarray返回转换为浮点类型的数组
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
print('Complete workout data ')

test_data_file = open("dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
Original = []
Detection = []
i = 1
for record in test_data_list:
    all_values = record.split(',')
    # 把10个数字显示出来
    plt.subplot(2, 5, i)
    plt.imshow(numpy.asfarray(all_values[1:]).reshape((28, 28)), cmap='Greys')
    plt.xticks([]), plt.yticks([])
    i += 1
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    Original.append(correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    Detection.append(label)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
# for i in range(len(test_data_list)):
#     plt.subplot(2, 5, i+1), plt.imshow(numpy.asfarray(all_values[1:]).reshape((28, 28)), cmap='Greys')


#计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
plt.suptitle('labels = '+str(Original)+'\n'+'result = '+str(Detection))
plt.show()
