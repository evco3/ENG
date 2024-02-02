import torch.nn as nn


class residual_block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels




    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()

        self.inplanes = 64

        # Initial convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create residual layers
        self.layers = []
        in_channels = 64
        for i, num_block in enumerate(num_blocks):
            out_channels = 64 * 2**i
            stride = 2 if i > 0 else 1
            layer = self._make_layer(residual_block, in_channels, out_channels, num_block, stride)
            self.layers.append(layer)
            in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * 2**(len(num_blocks)-1), num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





