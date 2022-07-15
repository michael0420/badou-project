import numpy as np
import matplotlib.pyplot as plt

[1]
'''
第一步是计算输入训练数据，给出网络的计算结果。（正向传播）
第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。（反向传播）
inputs_list:输入的训练数据;
targets_list:训练数据对应的正确结果。
'''
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
        hidden_errors = np.dot(self.who.T, outputs_errors*final_outputs*(1-final_outputs))
        # 根据误差更新权重
        self.who += self.lr*np.dot(outputs_errors*final_outputs*(1-final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs), np.transpose(inputs))
        pass


[2]
"""
使用实际数据来训练神经网络
"""

data_file = open("dataset/mnist_test.csv")
# data_file = open("E:/LearnImage/badouCode/lesson9/NeuralNetWork_from_zero/dataset/mnist_test.csv", "r")
data_list = data_file.readlines()
data_file.close()
# print(data_list)
print(len(data_list))
print(data_list[0])

'''
这里我们可以利用画图.py将输入绘制出来
从绘制的结果看，数据代表的确实是一个黑白图片的手写数字。
数据读取完毕后，我们再对数据格式做些调整，以便输入到神经网络中进行分析。
'''
#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')   # 字符串   通过指定分隔符对字符串进行切片
print(all_values)
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
# asfarrary不仅可以对多维数组做数值处理，还可以对列表做去转移符处理
print(image_array)
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()


[3]


'''
数据归一化处理
我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
'''

#数据预处理（归一化）
scaled_input = image_array / 255.0 * 0.99 + 0.01
# print(scaled_input)







