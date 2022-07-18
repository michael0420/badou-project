import torch
import torch.nn as nn
# 方法一：
model = nn.Sequential()
model.add_module("fc1", nn.linear(3, 4))
model.add_module("fc2", nn.linear(4, 2))
model.add_module("output", nn.Softmax(2))

# 方法二：

model2 = torch.nn.Sequential(nn.conv2d(1,20,5),
                       nn.Relu(),
                       nn.conv2d(20,64,5),
                       nn.Relu()
                       )

# 方法三：

model3 = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])

# 顺序执行时才可以写成ModuleList形式
