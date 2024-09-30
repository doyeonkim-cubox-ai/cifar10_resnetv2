import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def pick(m: str):
    if m == "resnet110":
        return resnet110()
    elif m == "v2resnet110":
        return v2resnet110()
    elif m == "resnet164":
        return resnet164()
    elif m == "v2resnet164":
        return v2resnet164()
    else:
        print(f"no such model exists: {m}")
        exit(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, opt=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)
        self.stride = stride
        self.opt = opt

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.opt != 0:
            identity = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.opt // 2, self.opt // 2), value=0)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, opt=0):
        super(BasicBlock2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*self.expansion, kernel_size=3, padding=1, bias=False)
        self.stride = stride
        self.opt = opt

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.opt != 0:
            identity = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.opt // 2, self.opt // 2), value=0)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, opt=0):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.stride = stride
        self.opt = opt
        self.downsample = None
        if self.opt != 0:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, bias=False, stride=stride),
                    nn.BatchNorm2d(planes*self.expansion)
                )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.opt != 0:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, opt=0):
        super(BottleNeck2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.stride = stride
        self.opt = opt
        self.downsample = None
        if self.opt != 0:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, bias=False, stride=stride),
                    nn.BatchNorm2d(planes*self.expansion)
                )

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.opt != 0:
            identity = self.downsample(x)

        out += identity
        return out


class ResNetV2(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetV2, self).__init__()

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layers(block, 16, layers[0])
        self.layer2 = self._make_layers(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layers(self, block, planes, blocks, stride=1):
        opt = 0
        if self.in_planes != planes*block.expansion or stride != 1:
            opt = self.in_planes
        layers = []
        layers.append(block(self.in_planes, planes, stride, opt))
        self.in_planes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)


def resnet110():
    return ResNetV2(BasicBlock, [18, 18, 18])


def v2resnet110():
    return ResNetV2(BasicBlock2, [18, 18, 18])


def resnet164():
    return ResNetV2(BottleNeck, [18, 18, 18])


def v2resnet164():
    return ResNetV2(BottleNeck2, [18, 18, 18])

