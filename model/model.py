import torch.nn as nn
from torchvision.models import resnet34


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convlayer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rconvlayer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        )

        self.convlayer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rconvlayer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        )

        self.convlayer7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.connlayer8 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.connlayer1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=2048),
            nn.LeakyReLU(),
        )

        self.connlayer2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=7 * 7 * 30),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.convlayer1(x)  # torch.Size([1, 64, 112, 112])

        x = self.convlayer2(x)  # torch.Size([1, 192, 56, 56])

        x = self.convlayer3(x)  # torch.Size([1, 512, 28, 28])

        for i in range(4):
            x = self.rconvlayer4(x)
        x = self.convlayer5(x)  # torch.Size([1, 1024, 14, 14])

        for i in range(2):
            x = self.rconvlayer6(x)
        x = self.convlayer7(x)  # torch.Size([1, 1024, 7, 7])

        # x=self.connlayer8(x)  # torch.Size([1, 1024, 7, 7])

        x = x.view(-1, 512 * 7 * 7)  # torch.Size([1, 1024*7*7])

        x = self.connlayer1(x)  # torch.Size([1, 4096])

        x = self.connlayer2(x)  # torch.Size([1, 1470])

        x = x.view(-1, 30, 7, 7)

        return x

