"""Wide_ResNet in PyTorch.

Reference:
[1] Sergey Zagoruyko, Nikos Komodakis
    Wide Residual Networks. https://arxiv.org/abs/1605.07146
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class wide_basic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    def __init__(self, block, depth, widen_factor, dropout_rate, num_classes=10, in_channels=3, num_out_conv1=16):
        super(Wide_ResNet, self).__init__()
        self.in_planes = num_out_conv1

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n + 4.'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [
            num_out_conv1,
            1 * k * num_out_conv1,
            2 * k * num_out_conv1,
            4 * k * num_out_conv1,
        ]

        self.conv1 = nn.Conv2d(in_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(block, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(block, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def WRN_40_1(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 40, 1, dropout_rate, num_classes, in_channels)


def WRN_40_2(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 40, 2, dropout_rate, num_classes, in_channels)


def WRN_40_4(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 40, 4, dropout_rate, num_classes, in_channels)


def WRN_40_8(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 40, 8, dropout_rate, num_classes, in_channels)


def WRN_28_10(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 28, 10, dropout_rate, num_classes, in_channels)


def WRN_28_12(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 28, 12, dropout_rate, num_classes, in_channels)


def WRN_22_8(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 22, 8, dropout_rate, num_classes, in_channels)


def WRN_22_10(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 22, 10, dropout_rate, num_classes, in_channels)


def WRN_16_8(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 16, 8, dropout_rate, num_classes, in_channels)


def WRN_16_10(dropout_rate=0.3, num_classes=10, in_channels=3):
    return Wide_ResNet(wide_basic, 16, 10, dropout_rate, num_classes, in_channels)


def test():
    net = WRN_28_10()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
