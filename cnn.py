import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config):
        super(FlexibleCNN, self).__init__()
        self.config = config
        self._configure_model()

    def _configure_model(self):
        if self.config == 1:
            self.conv1 = nn.Conv2d(1, 64, 5, padding=2) 
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 128)  
            self.fc2 = nn.Linear(128, 10)  
        elif self.config == 2:
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  
            self.fc1 = nn.Linear(128 * 4 * 4, 128) 
            self.fc2 = nn.Linear(128, 10)  
        elif self.config == 3:
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 4 * 4, 256)  
            self.fc2 = nn.Linear(256, 128)  
            self.fc3 = nn.Linear(128, 10) 
        else:
            raise ValueError("Invalid config, only 1, 2, or 3 allowed.")

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        if self.config == 1:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.config == 2:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.config == 3:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x
