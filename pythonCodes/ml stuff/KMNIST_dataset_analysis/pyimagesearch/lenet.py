from torch import conv2d
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import Flatten

class LeNet(Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()
        
        self.con2D1 = conv2d(in_channels = numChannels, out_channels = 20 kernel_size = (5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.convsD2 = Conv2d(in_channels = 20, out_channels = 50, kernel_size = (5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.fc1 = Linear(in_features = 50*4*4, out_features = 500)
        self.relu3 = ReLU()
        
        self.fc2 = Linear(in_features = 500, out_features = classes)
        self.logSoftmax = LogSoftmax(dim = 1)