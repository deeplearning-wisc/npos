import torch.nn as nn
import torch
import torch.nn.functional as F

class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes = 80):
        super(clssimp, self).__init__()
        # self.way1 = nn.Sequential(
        #     nn.Linear(ch, 1024, bias=True),
        #     nn.GroupNorm(num_channels=1024, num_groups=32),
        #     # nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        # )
        # self.cls= nn.Linear(2048, num_classes,bias=True)

        # self.conv = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
        self.conv = nn.Conv2d(in_channels= ch, out_channels=num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # beta = 0
        x = self.conv(x) # 64x9x7x7
        # max_x = F.adaptive_max_pool2d(x, (1,1))  
        # x = x * (x > beta*max_x)     
        x = self.pool(x) # 64x9x1x1
        logits = x.reshape(x.size(0), -1)  #64x9
        return logits, 

    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        return x

