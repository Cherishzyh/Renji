import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def Conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def Conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1x1x1 = Conv1x1x1(planes+1, planes)
        self.conv1 = Conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        residual = inputs[0]
        atten_map = inputs[1]


        out = self.conv1(inputs[0])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shape = out.shape[2:]
        atten_map = torch.nn.functional.interpolate(atten_map, size=shape, mode='trilinear', align_corners=True)

        out = torch.cat([out, atten_map], dim=1)
        out = self.conv1x1x1(out)

        if self.downsample is not None:
            residual = self.downsample(inputs[0])

        out += residual
        out = self.relu(out)

        return [out, inputs[1]]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1x1x1 = Conv1x1x1(planes * self.expansion + 1, planes * self.expansion)
        self.conv1 = Conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = Conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = Conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        residual = inputs[0]
        atten_map = inputs[1]

        out = self.conv1(inputs[0])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shape = out.shape[2:]
        atten_map = torch.nn.functional.interpolate(atten_map, size=shape, mode='trilinear', align_corners=True)

        out = torch.cat([out, atten_map], dim=1)
        out = self.conv1x1x1(out)

        if self.downsample is not None:
            residual = self.downsample(inputs[0])

        out += residual
        out = self.relu(out)

        return [out, inputs[1]]


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_classes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion,
                                           block_inplanes[0] * block.expansion),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(block_inplanes[0] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    Conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(in_planes=self.in_planes,
                            planes=planes,
                            stride=stride,
                            downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, atten_map):
        x = self.conv1(x)                #(batch, 64,  30,  100, 100)
        x = self.bn1(x)                  #(batch, 64,  30,  100, 100)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)          #(batch, 64,  15,  50, 50)

        x = self.layer1([x, atten_map])               #(batch, 256,  15, 50, 50)
        x = self.layer2(x)               #(batch, 512,  8,  25, 25)
        x = self.layer3(x)               #(batch, 1024, 4,  13, 13)
        x = self.layer4(x)               #(batch, 2048, 2,  7,  7)

        x = self.avgpool(x[0])              #(batch, 2048, 1,  1,  1)

        x = x.view(x.size(0), -1)        #(batch, 2048)
        x = self.fc2(self.fc1(x))                   #(batch, n_classes)

        return x


def GenerateModel(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


if __name__ == '__main__':
    # model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=1, conv1_t_size=7, conv1_t_stride=1,
    #              no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=4)
    model = GenerateModel(50, n_input_channels=1, n_classes=4)
    input_image = torch.from_numpy(np.random.rand(3, 1, 30, 200, 200)).float()
    pred = model(input_image, input_image)
    print(pred.shape)