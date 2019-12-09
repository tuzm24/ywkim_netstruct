import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import os

class test1(nn.Module):
    ch = 64
    def __init__(self):
        super(test1, self).__init__()
        self.conv1 = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):

        x  = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class test2(nn.Module):
    ch = 64
    def __init__(self):
        super(test2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=False))
    def forward(self,x):
        x = self.features(x)
        return x

def test4(r = os.path.splitext(os.path.basename(__file__))[0]):
    print(r)

if '__main__' == __name__:
    net = test1()
    summary(net, (64,3840,2160), device='cuda')
    net2 = test2()
    summary(net2, (64,3840,2160), device='cuda')