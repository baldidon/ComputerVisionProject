import torch
import torch.nn as nn

class Net(nn.Module):
    # selector for CIFAR100 or CIFAR10
    def __init__(self, selector):
        super(Net, self).__init__()
        self.selector = selector

        #remember. CIFAR images: 32*32*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=(3,3),
                      padding=(1,1),# padding same
                      stride=(1,1)),
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
            nn.MaxPool2d(2) #16*16*128
        )

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=(1,1), padding=(0,0),stride=(1,1)),
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
            nn.MaxPool2d(2) #8*8*256
        )

        self.conv5 = nn.Sequential(
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
                      out_channels=128,
                      kernel_size=(1,1),
                      padding=(0,0),
                      stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(8*8*128, 100)

        if(selector == 10):
            self.fc2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(100,10)
            )



    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        bottleneck1= self.bottleneck1(conv3)
        conv4 = self.conv4(bottleneck1)
        conv5 = self.conv5(conv4)
        bottleneck2 = self.bottleneck2(conv5)
        fc = self.fc(bottleneck2.view(bottleneck2.size(0),-1))

        if self.selector == 10:
            fc2 = self.fc2(fc)
            return fc2

        return fc
