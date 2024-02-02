import torch
import nnsvg
from torchvision import models

# Define your ResNet model here (the code you provided)
class residual_block(nn.Module):
    def convBatchNorm(self, input_size, output_size, kernel_size):
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
        )

    def __init__(self, input, output_size, stride=1):
        super(residual_block, self).__init__()
        self.conv1 = self.convBatchNorm(input, output_size, 3)
        self.conv2 = self.convBatchNorm(output_size, output_size, 3)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if input != output_size or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input, output_size,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_size)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut_output = self.shortcut(x)
        if shortcut_output.shape == out.shape:
            out += shortcut_output
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.input_size = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.form_layer(64, blocks[0], stride=1)
        self.layer2 = self.form_layer(128, blocks[1], stride=2)
        self.layer3 = self.form_layer(256, blocks[2], stride=2)
        self.layer4 = self.form_layer(512, blocks[3], stride=2)
        self.linear = nn.Sequential(
            nn.Linear(32768, 512 * 4 * 4),
            nn.Linear(512 * 4 * 4, num_classes)
        )

    def form_layer(self, size, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(residual_block(self.input_size, size, stride))
            self.input_size = size
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# Create an instance of your ResNet model
model = ResNet([2, 2, 2, 2])

# Create an instance of the NN-SVG drawer
drawer = nnsvg.Drawer()

# Create a random input tensor to determine the input size
sample_input = torch.randn(1, 3, 32, 32)  # Adjust the input size as needed

# Draw the model and save the diagram as an SVG file
drawer.draw(model, sample_input)
drawer.save("resnet_diagram.svg")
