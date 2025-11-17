import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super(DQN, self).__init__()
        # "The input is an 84x84x4 image"
        # "The first hidden layer convolves 16 filters of 8x8 with stride 4"
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        # "Applies a rectifier nonlinearity"
        # "The second hidden layer convolves 32 filters of 4x4 with stride 2 + ReLU"
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        # "The final hidden layer is fully-connected with 256 units + ReLU"
        self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=n_actions)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
