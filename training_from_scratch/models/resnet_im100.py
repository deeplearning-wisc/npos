import torch.nn as nn
import torch
from torchvision import models
class resnet101(nn.Module):
    def __init__(self, num_class=100):
        super(resnet101, self).__init__()
        self.model = models.resnet101(pretrained=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )
        self.outlier_MLP = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.fc = nn.Linear(512, num_class)


    def forward(self, x, fc=True, mlp=False):
        if mlp==True:
            return self.outlier_MLP(x)
        if fc==False:
            batch = x.size(0)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            # print(x.shape)
            #feature = x.view(batch, -1)
            #feature = self.proj(feature)
            #logit = self.fc(feature)
            return x
        batch = x.size(0)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        #print(x.shape)
        feature = x.view(batch, -1)
        feature = self.proj(feature)
        logit = self.fc(feature)
        return logit, x, feature
