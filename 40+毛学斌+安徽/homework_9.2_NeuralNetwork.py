import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化网络，设置输入层、隐藏层、输出层的节点数，以及学习率
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lrate = learning_rate
        # 初始化权重矩阵：权重w，i2h输入到隐藏层，因为h=w*i，所以矩阵为（h，i），-0.5调整到均值为0，正负都有
        self.w_i2h = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.w_h2o = np.random.rand(self.onodes, self.hnodes) - 0.5
        # 激活函数，后面的就是sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_in, targets_in):  # 训练
        inputs_in = np.array(inputs_in, ndmin=2).T  # 一维转二维后是行向量，加转置改为列向量便于后面计算
        targets_in = np.array(targets_in, ndmin=2).T
        # 正向传播
        hidden_input = np.dot(self.w_i2h, inputs_in)  # 加权求和
        hidden_output = self.activation_function(hidden_input)  # 激活函数
        final_input = np.dot(self.w_h2o, hidden_output)
        final_output = self.activation_function(final_input)
        # 反向传播
        output_errors = final_output - targets_in  # 参照上一节ppt的推导公式
        hidden_errors = np.dot(self.w_h2o.T, output_errors * final_output * (1 - final_output))
        # 以上用到矩阵的左除 w*A=B,则A = w\B = w的逆矩阵*B 。 同时后面权重矩阵会更新，要先算
        self.w_h2o -= self.lrate * np.dot((output_errors * final_output *
                                          (1 - final_output)), hidden_output.T)
        # 以上用到矩阵的右除 w*A=B,则w = B/A = B*A的逆矩阵。通常用转置替代逆矩阵。因为逆矩阵不一定能求出来。
	# 目前阶段只有方阵才能救逆矩阵。本步的目的是收敛，而非绝对值的大小。采用转置可以保证计算能继续。
        self.w_i2h -= self.lrate * np.dot((hidden_errors * hidden_output *
                                          (1 - hidden_output)), inputs_in.T)

    def predict(self, inputs):  # 预测,正向传播
        hidden_in = np.dot(self.w_i2h, inputs)  # 输入层到隐藏层加权求和
        hidden_out = self.activation_function(hidden_in)  # 激活函数
        final_in = np.dot(self.w_h2o, hidden_out)  # 隐藏层到输出层加权求和
        final_out = self.activation_function(final_in)
        # print('final_out:', final_out)
        return final_out


# 1、设置参数，并用类创造1个对象
nodes_input = 784  # 手写数字的图片大小为28，28，计算时需要转成一维
nodes_hidden = 200  # 人为设定
nodes_output = 10  # 输出0-9共10个数
rate = 0.5  # 学习率
network = NeuralNetwork(nodes_input, nodes_hidden, nodes_output, rate)

# 2、读入测试数据
train_data = open('mnist_train.csv', 'r')  # 导入数据
train_data_list = train_data.readlines()  # 逐行读入。注意有s，readlines
train_data.close()  # open打开的都要close关闭
print(type(train_data_list))  # 数字类型为字符串
print('本次训练数据条数为：', len(train_data_list))
# print('train_data_list[0]:', train_data_list[0])  # 每一行是一个字符串

# 3、训练
epochs = 5  # 迭代次数，重复几遍
for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(',')  # 去掉，将字符串转化成列表
        input_train = (np.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
        # 先转化为浮点型数组as f array, 再归一化，+0.01为了避免值太小导致的0计算失真
        targets = np.zeros(nodes_output) + 0.01  # 加0.01目的如上
        targets[int(all_values[0])] = 0.99
        network.train(input_train, targets)

# 4、读入验证数据并进行验证
test_data = open('mnist_test.csv', 'r')
test_data_list = test_data.readlines()
test_data.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    input_test = (np.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
    result = network.predict(input_test)
    target = int(all_values[0])  # 将字符串转化成数字便于后面比较用
    print('本次检测数字为：', target)
    print('本次预测数字为：', np.argmax(result))
    print(result)
    if target == np.argmax(result):
        scores.append(1)
    else:
        scores.append(0)
scores_array = np.array(scores)
accuracy = scores_array.sum() / scores_array.size
print('预测正确率为：', accuracy)
