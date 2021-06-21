import torch
import torch.nn as nn
from nets.darknet import DarkNet


class YOLOv1(nn.Module):
    def __init__(self, features, num_bboxes=2, num_classes=20, bn=True):
        super(YOLOv1, self).__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.features = features
        self.conv_layers = self._make_conv_layers(bn)
        self.fc_layers = self._make_fc_layers()

    @staticmethod
    def _make_conv_layers(bn):
        if bn:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
                nn.LeakyReLU(0.1, inplace=True),
                # nn.Conv2d(1024, 1024, 3, padding=1),
                # nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
                # nn.LeakyReLU(0.1, inplace=True)
            )

        else:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        return net

    def _make_fc_layers(self):
        s, b, c = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 2048),  # 7 * 7 * 1024, 4096
            nn.LeakyReLU(0.1, inplace=True),
            #             nn.Dropout(0.5, inplace=False),  # is it okay to use Dropout with BatchNorm?
            nn.Linear(2048, s * s * (5 * b + c)),  # 4096, s * s * (5 * b + c)
            nn.Sigmoid()
        )

        return net

    def forward(self, x):
        s, b, c = self.feature_size, self.num_bboxes, self.num_classes

        x = self.features(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        x = x.view(-1, s, s, 5 * b + c)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    # Build model with randomly initialized weights
    darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
    yolo = YOLOv1(darknet.features)

    # Dummy image
    image = torch.randn(2, 3, 448, 448)  # torch.Size([2, 3, 448, 448])

    output = yolo(image)

    print(output.size())  # torch.Size([2, 7, 7, 30])


if __name__ == '__main__':
    main()
