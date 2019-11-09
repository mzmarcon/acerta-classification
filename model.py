import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class VGGBased13(nn.Module):
    def __init__(self):
        super(VGGBased13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(61, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            # nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.Dropout2d(),
            #
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            # nn.Dropout2d(),
            #
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            #
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.Dropout2d(),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.classifier = nn.Sequential(
            # nn.Linear(512, 2048),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            # nn.BatchNorm1d(),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]))

        return x


class VGGBased4(nn.Module):
    def __init__(self):
        super(VGGBased2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            # nn.Dropout2d(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            # nn.Dropout2d(),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),

        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 2048),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1),
            # nn.BatchNorm1d(),
            # nn.ReLU()
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # print(x.shape)

        # return F.Sigmoid(x)
        return x