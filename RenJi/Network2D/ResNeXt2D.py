import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, input_channels, num_blocks, cardinality=32, bottleneck_width=4, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(cardinality*bottleneck_width*16,
                                           cardinality*bottleneck_width*4),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(cardinality*bottleneck_width*4, num_classes)
        # self.linear = nn.Linear(int(cardinality*bottleneck_width*math.pow(2, len(num_blocks))), num_classes)


    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # [1, 64, 100, 100]
        out = self.maxpool(out)                 # [1, 64, 50, 50]
        out = self.layer1(out)                  # [1, 256, 50, 50]
        out = self.layer2(out)                  # [1, 512, 25, 25]
        out = self.layer3(out)                  # [1, 1024, 13, 13]
        out = self.layer4(out)                  # [1, 2048, 7, 7]
        out = self.avgpool(out)                 # [1, 2048, 1, 1]
        out = out.view(out.size(0), -1)         # [1, 2048]
        out = self.fc2(self.fc1(out))           # [1, 4]
        return out


if __name__ == '__main__':
    net = ResNeXt(input_channels=3, num_classes=4, num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)
    print(net)
    x = torch.randn(1, 3, 200, 200)
    y = net(x)
    print(y.size())

