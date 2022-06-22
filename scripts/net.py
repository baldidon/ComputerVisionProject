import torch
import torch.nn as nn

class Net(nn.Module):
    # selector for CIFAR100 or CIFAR10
    # transfer_learning: if 1 use parameters of opposite CIFAR
    def __init__(self, CIFAR, max_pool):
        super(Net, self).__init__()
        self.selector = CIFAR

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
        
        layers = list()
        if max_pool == 1:
            layers = [
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          padding=(1, 1),  # padding same
                          stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2) 
            ]
        else:
            layers = [nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          padding=(1, 1),  # padding same
                          stride=(2, 2)),
                nn.BatchNorm2d(128),
                nn.ReLU()
                    ]

        
        self.conv3 = nn.Sequential(*layers) #16*16*128

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=(1,1), padding=(0,0),stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        layers2 = list()
        if max_pool == 1:
            layers2 = [
                nn.Conv2d(in_channels=64,
                          out_channels=256,
                          kernel_size=(3, 3),
                          padding=(1, 1),  # padding same
                          stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2) 
            ]
        else:
            layers2 = [nn.Conv2d(in_channels=64,
                          out_channels=256,
                          kernel_size=(3, 3),
                          padding=(1, 1),  # padding same
                          stride=(2, 2)),
                nn.BatchNorm2d(256),
                nn.ReLU()
                    ]
        
        self.conv4 = nn.Sequential(*layers2) #8*8*256

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      padding=(1, 1),  # padding same
                      stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.fc = nn.Linear(8*8*256, self.selector)
        


    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        bottleneck1 = self.bottleneck1(conv3)
        conv4 = self.conv4(bottleneck1)
        conv5 = self.conv5(conv4)
        fc = self.fc(conv5.view(conv5.size(0), -1))
        return fc
