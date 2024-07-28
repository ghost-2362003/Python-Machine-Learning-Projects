from torch import conv2d, flatten
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax

class LeNet(Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()
        
        self.con2D1 = conv2d(in_channels = numChannels, out_channels = 20, kernel_size = (5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv2D2 = Conv2d(in_channels = 20, out_channels = 50, kernel_size = (5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.fc1 = Linear(in_features = 50*4*4, out_features = 500)
        self.relu3 = ReLU()
        
        self.fc2 = Linear(in_features = 500, out_features = classes)
        self.logSoftmax = LogSoftmax(dim = 1)
        
    def forward(self, x):
        x = self.conv2D1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2D2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.logSoftmax(x)
        
        return x
