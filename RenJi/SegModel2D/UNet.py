import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = DoubleConv(in_channels, filters)
        self.conv2 = DoubleConv(filters, filters*2)
        self.conv3 = DoubleConv(filters*2, filters*4)
        self.conv4 = DoubleConv(filters*4, filters*8)
        self.conv5 = DoubleConv(filters*8, filters*16)

        self.up6 = nn.ConvTranspose2d(filters*16, filters*8, 2, stride=2)
        self.conv7 = DoubleConv(filters*16, filters*8)

        self.up7 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv8 = DoubleConv(filters*8, filters*4)

        self.up8 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv9 = DoubleConv(filters*4, filters*2)

        self.up9 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv10 = DoubleConv(filters*2, filters)

        self.conv11 = nn.Conv2d(filters, out_channels, 1)

    def _padding(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x1


    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)

        x4 = self.pool(x3)
        x4 = self.conv4(x4)

        x5 = self.pool(x4)
        x5 = self.conv5(x5)

        up_6 = self.up6(x5)
        up_6 = self._padding(up_6, x4)
        merge1 = torch.cat((up_6, x4), dim=1)
        x6 = self.conv7(merge1)

        up_7 = self.up7(x6)
        up_7 = self._padding(up_7, x3)
        merge2 = torch.cat((up_7, x3), dim=1)
        x7 = self.conv8(merge2)

        up_8 = self.up8(x7)
        up_8 = self._padding(up_8, x2)
        merge3 = torch.cat((up_8, x2), dim=1)
        x8 = self.conv9(merge3)

        up_9 = self.up9(x8)
        up_9 = self._padding(up_9, x1)
        merge4 = torch.cat((up_9, x1), dim=1)
        x9 = self.conv10(merge4)

        x10 = self.conv11(x9)
        return x10


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=5)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 1, 184, 184).to(device)
    prediction = model(inputs)
    print(prediction.shape)


if __name__ == '__main__':

    test()