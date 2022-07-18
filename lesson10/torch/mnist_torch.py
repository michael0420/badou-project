import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
'''
tochvision主要处理图像数据，包含一些常用的数据集、模型、转换函数等。torchvision独立于PyTorch，需要专门安装。
torchvision主要包含以下四部分：
torchvision.models: 提供深度学习中各种经典的网络结构、预训练好的模型，如：Alex-Net、VGG、ResNet、Inception等。
torchvision.datasets：提供常用的数据集，设计上继承 torch.utils.data.Dataset，主要包括：MNIST、CIFAR10/100、
ImageNet、COCO等。
torchvision.transforms：提供常用的数据预处理操作，主要包括对Tensor及PIL Image对象的操作。
torchvision.utils：工具类，如保存张量作为图像到磁盘，给一个小批量创建一个图像网格。
'''


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.creat_cost(cost)
        self.optimizer = self.creat_optimizer(optimist)
        pass

    def creat_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def creat_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]


    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):   # enumerate返回的是元素以及对应的索引。
                inputs, labels = data

                self.optimizer.zero_grad()  # 梯度归零
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)  # 计算损失函数
                loss.backward()  # 反向传播
                self.optimizer.step()  # 参数更新

                running_loss += loss.item()
                if i % 100 == 0:  # 每100个batch数据计算一次损失平均值
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0   # 重新将running_loss置0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。因为测试不需要更新权值。
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100*correct / total))   # %% 表示字符"%"


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),  # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
         transforms.Normalize([0,], [1,])])
    # ToTensor()  能够把灰度范围从0 - 255变换到0 - 1之间，
    # transform.Normalize()  表示归一化操作  用均值和标准差归一化张量图像
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    #     shuffle=True,               # 要不要打乱数据 (打乱比较好)
    #     num_workers=2,              # 多线程来读数据
    return trainloader, testloader
    # CLASS   torchvision.datasets.MNIST(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform:
    # Optional[Callable] = None, download: bool = False)
    # root(string)： 表示数据集的根目录，其中根目录存在MNIST / processed / training.pt和MNIST / processed / test.pt的子目录
    # train(bool, optional)： 如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
    # download(bool, optional)： 如果为True，则从internet下载数据集并将其放入根目录。如果数据集已下载，则不会再次下载
    # transform(callable, optional)： 接收PIL图片并返回转换后版本图片的转换函数
    # target_transform(callable, optional)： 接收PIL接收目标并对其进行变换的转换函数


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()   #  super 的一个最常见用法可以说是在子类中调用父类的初始化方法了
        self.fc1 = torch.nn.Linear(28*28, 512)   # 设置网络结构
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # dim指的是归一化的方式，如果为0是对列做归一化，1是对行做归一化。
        return x


if __name__ =='__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)

