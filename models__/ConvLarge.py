import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.utils import weight_norm
from torch.autograd.variable import Variable
import math

__all__ = ['convlarge']


# noise function taken from blog : https://ferretj.github.io/ml/2018/01/22/temporal-ensembling.html?fbclid=IwAR1MEqzhwrl1swzLUDA0kZFN2oVTdcNa497c1l3pC-Xh2kYPlPjRiO0Oucc
class GaussianNoise(nn.Module):

    def __init__(self, shape=(100, 1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.noise1 = Variable(torch.zeros(shape).cuda())
        self.std1 = std
        self.register_buffer('noise2',self.noise1) # My own contribution , registering buffer for data parallel usage

    def forward(self, x):
        c = x.shape[0]
        self.noise2.data.normal_(0, std=self.std1)
        return x + self.noise2[:c]

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()      
        self.info = 'in ' +  str(in_channels) + ', out ' + str(out_channels) + ', stride ' + str(stride)
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, 0, out_channels - in_channels), mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
 
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        bn2 = nn.BatchNorm2d(out_channels)
        self.l1 = nn.Sequential(conv1, bn1)
        self.l2 = nn.Sequential(conv2, bn2)

    def forward(self, input):
        skip = self.skip(input)
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        return F.relu(x + skip)

class ResidualStack(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        first = [ResidualBlock(in_channels, out_channels, stride)]
        rest = [ResidualBlock(out_channels, out_channels) for i in range(num_blocks - 1)]
        self.modules_list = nn.Sequential(*(first + rest))
        
    def forward(self, input):
        return self.modules_list(input)

class Net(nn.Module):
    def __init__(self,args,std = 0.05):
        super(Net, self).__init__()
        self.args = args

        self.std = std
        self.gn = GaussianNoise(shape=(args.batch_size,3,28,28),std=self.std)
        

        self.res_net_18 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResidualStack(64, 64, 1, 6),
                ResidualStack(64, 128, 2, 8),
                ResidualStack(128, 256, 2, 12),
                ResidualStack(256, 512, 1, 6),
                nn.AdaptiveAvgPool2d(1),
                Lambda(lambda x: x.squeeze()),
                nn.Linear(512, 9)
            )


    def forward(self, x, ema_input=False):

        if self.training and True:#ema_input:
             x = self.gn(x)
        
        return self.res_net_18(x)

        if self.args.sntg == True:
            return x,h
        else:
            return x


def convlarge(args,data= None,nograd=False):
    model = Net(args)
    if data is not None:
        model.load_state_dict(data['state_dict'])



    model = model.cuda()
    model = nn.DataParallel(model).cuda()

    if nograd:
        for param in model.parameters():
            param.detach_()


    return model
