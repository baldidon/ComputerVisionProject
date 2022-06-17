import torch
import torch.nn as nn


class Net(nn.Module):
    # selector for CIFAR100 or CIFAR10
    # transfer_learning: if 1 use parameters of opposite CIFAR
    def __init__(self, CIFAR, transfer_learining, dropout):
        super(Net, self).__init__()
        self.selector = CIFAR

        # remember. CIFAR images: 32*32*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=128,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16*16*128
        )

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8*8*256
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        if transfer_learining == False:
            self.fc2 = nn.Linear(512, self.selector)
        elif self.selector == 100:
            self.fc2 = nn.Linear(512, 10)
        else:
            self.fc2 = nn.Linear(512, 100)
        # swap for a moment fully-connected layers! We need to do that for correct
        # loading of parameters!


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        bottleneck1 = self.bottleneck1(conv3)
        conv4 = self.conv4(bottleneck1)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        bottleneck2 = self.bottleneck2(conv6)
        fc1 = self.fc(bottleneck2.view(bottleneck2.size(0), -1))
        fc2 = self.fc2(fc1)
        return fc2
