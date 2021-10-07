import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, conv_only=False, bn=True, init_weight=True):
        super(Backbone, self).__init__()

        # Make layers
        self.features = self._make_conv_bn_layers() if bn else self._make_conv_layers()
        if not conv_only:
            self.fc = self._make_fc_layers()

        # Initialize weights
        if init_weight:
            self._initialize_weights()

        self.conv_only = conv_only

    def forward(self, x):
        x = self.features(x)
        if not self.conv_only:
            x = self.fc(x)
        return x

    @staticmethod
    def _make_conv_bn_layers():
        conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(192, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(192, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv

    @staticmethod
    def _make_conv_layers():
        conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(192, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv

    @staticmethod
    def _make_fc_layers():
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 1000)
        )
        return fc

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


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


def main():
    # Build model
    model = Backbone().features

    # Dummy image
    image = torch.randn(2, 3, 448, 448)

    output = model(image)

    print(output.shape)  # torch.Size([2, 1024, 14, 14])


if __name__ == '__main__':
    main()
