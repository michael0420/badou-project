import torch


#  linear torch的内部实现，用的时候直接调用接口就行
class linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))


    def forward(self, x):
        x = x.mm(self.weight)
        if self.bias:
            x= x + self.bias.expand_as(x)

        return(x)


if __name__ == '__main__':
    net = linear(3, 2)
    x = net.forward
    print('11', x)
