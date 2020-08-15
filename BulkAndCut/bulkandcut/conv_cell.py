import numpy as np
import torch


class ConvCell(torch.nn.Module):

    def __init__(self, in_channels, rng):
        super(ConvCell, self).__init__()

        # Sample architecture
        self.out_channels = rng.integers(low=400, high=800)
        self.kernel_size = rng.choice([3, 5, 7])
        self.in_channels = in_channels
        self.padding = (self.kernel_size - 1) // 2

        self.conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            )
        self.drop = torch.nn.Dropout2d(p=0.5)
        self.act = torch.nn.ReLU()
        self.bnorm = torch.nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.bnorm(x)
        return x

