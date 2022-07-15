import torch
from torch import nn
import numpy as np
from torch.autograd import Variable



class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride='same',padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(256*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)

        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,256*6*6)
        x = self.classifier(x)
        return x