import numpy as np
import matplotlib.pyplot as plt

[1]
"""
读取训练数据
"""
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

# 把数据依靠','区分，并依次读入
all_values =data_list[0].split(',')
# 第一个值对应的是图片表示的数字，读取图片数据时要去掉第一个数值
image_array = np.asfarray(all_values[1:]).reshape((28, 28))


# 设定输出节点数  最外层有10个输出节点
onodes = 10
targets = np.zeros(onodes) +0.01
targets[int(all_values[0])] = 0.99
print(targets)   #  [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.99 0.01 0.01]
                 # targets第8个元素的值为0.99，说明图片对应的数字是7。(数组从0开始编号)


[2]
'''
根据上述做法，我们就能把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
这里需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
'''

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
# 把数据依靠','区分开，并依次读入
for record in train_data_list:
    all_values =record.split(',')
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    targets=np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)


[3]

"""
最后把所有测试图片都输入网络，看看检测效果
"""
epochs = 10
for e in range(epochs):
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
            scores = scores.append(1)
        else:
            scores = scores.append(0)
    print(scores)

    # 计算图片判断的成功率
    scores_array = np.asarray(scores)  # 将列表转化为数组
    print("performance: = ", scores_array.sum()/scores_array.size)   #numpy.size(a, axis=None)  array.size不指定参数是返回数组元素个数
