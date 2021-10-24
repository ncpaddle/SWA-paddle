"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import paddle
import paddle.nn as nn
from paddle.vision import transforms

__all__ = ['VGG16', 'VGG16BN', 'VGG19', 'VGG19BN']


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1, weight_attr=nn.initializer.KaimingNormal())
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Layer):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512, bias_attr=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512, bias_attr=True),
            nn.ReLU(True),
            nn.Linear(512, num_classes, bias_attr=True),
        )


    def forward(self, x):
        x = self.features(x)

        x = paddle.reshape(x, [x.shape[0], -1])
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VGG16(Base):
    pass


class VGG16BN(Base):
    kwargs = {'batch_norm': True}


class VGG19(Base):
    kwargs = {'depth': 19}


class VGG19BN(Base):
    kwargs = {'depth': 19, 'batch_norm': True}
