"""
@author: Yakhyokhuja Valikhujaev <yakhyo9696@gmail.com>
"""

import torch
import torch.nn as nn
from utils.utils import GlobalAvgPool2d, Flatten
from typing import Optional


def auto_pad(kernel_size: int, padding: int):
    if padding is None:
        padding = kernel_size // 2
    return padding


class Conv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            dilation: Optional[int] = 1,
            groups: Optional[int] = 1,
            act: Optional[bool] = True
    ) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size, padding),
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1e-3)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class BACKBONE(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True):
        super(BACKBONE, self).__init__()

        self.features = nn.Sequential(
            Conv(3, 64, 7, 2),
            nn.MaxPool2d(2, 2),

            Conv(64, 192, 3),
            nn.MaxPool2d(2, 2),

            Conv(192, 128, 1),
            Conv(128, 256, 3),
            Conv(256, 256, 1),
            Conv(256, 512, 3),
            nn.MaxPool2d(2, 2),

            Conv(512, 256, 1),
            Conv(256, 512, 3),

            Conv(512, 256, 1),
            Conv(256, 512, 3),

            Conv(512, 256, 1),
            Conv(256, 512, 3),

            Conv(512, 256, 1),
            Conv(256, 512, 3),

            Conv(512, 512, 1),
            Conv(512, 1024, 3),
            nn.MaxPool2d(2, 2),

            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3)
        )

        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class HEAD(nn.Module):
    def __init__(self, fs, nb, nc):
        super(HEAD, self).__init__()

        self.conv = nn.Sequential(
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3, 2),
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3)
        )

        self.detect = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),  # 7 * 7 * 1024, 4096
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, fs * fs * (5 * nb + nc)),  # 4096, s * s * (5 * b + c)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.detect(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, fs=7, nb=2, nc=20, pretrained_backbone=False):
        super(YOLOv1, self).__init__()

        self.FS = fs
        self.NB = nb
        self.NC = nc
        if pretrained_backbone:

            self.features = BACKBONE().features
            darknet = BACKBONE()
            darknet = nn.DataParallel(darknet)
            src_state_dict = torch.load('model_best.pth.tar')['state_dict']
            dst_state_dict = darknet.state_dict()

            for k in dst_state_dict.keys():
                print('Loading weight of', k)
                dst_state_dict[k] = src_state_dict[k]
            darknet.load_state_dict(dst_state_dict)
            self.features = darknet.module.features
        else:
            self.features = BACKBONE().features
        self.head = HEAD(self.FS, self.NB, self.NC)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)

        x = x.view(-1, self.FS, self.FS, 5 * self.NB + self.NC)
        return x


if __name__ == '__main__':
    yolo = YOLOv1()

    # Dummy image
    image = torch.randn(2, 3, 448, 448)  # torch.Size([2, 3, 448, 448])

    output = yolo(image)

    print(output.size())  # torch.Size([2, 7, 7, 30])
