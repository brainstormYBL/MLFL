"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Defining the different models used to train different dataset
"""
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LinearRegression, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.network = nn.Sequential(nn.Linear(self.dim_input, 360),
                                     nn.Linear(360, self.dim_output))

    def forward(self, x):
        output = self.network(x)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 360, 5, 1)
        self.conv2 = nn.Conv2d(360, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)
