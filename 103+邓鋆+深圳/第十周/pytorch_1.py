import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam


class MnistModel(nn.Module):

    def __init__(self):
        super(MnistModel, self).__init__()  # 继承
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)


    def forward(self,x):
        #x = x.view(-1, 28 * 28)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.softmax(self.fc3(x), dim=1)
        #return x
        x = x.view(-1,28*28)#-1表示根据形状自动调整，也可以改为input.size(0)
       ##2.进行全连接的操作
        x = self.fc1(x)
        #3.激活函数的处理
        x = F.relu(x)#形状没有变化

        x = self.fc2(x)
        x = F.relu(x)  # 形状没有变化

        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def train(net,train_loader, optimizer,epoches=1):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                #loss = cost(outputs, labels)
                loss = nn.CrossEntropyLoss()
                loss = loss(outputs, labels)
                #loss = F.nll_loss(outputs, labels)  # 调用损失函数，得到损失,是一个tensor
                loss.backward()
                optimizer.step()

                #print('loss ',loss)
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

def evaluate(net,test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))



def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader




if __name__ == '__main__':
    train_loader, test_loader = mnist_load_data()

    net = MnistModel()
    optimizer = Adam(net.parameters(), lr=0.001)
    train(net,train_loader,optimizer)
    evaluate(net,test_loader)