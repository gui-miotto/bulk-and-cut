import numpy as np
import torch


class LinearCell(torch.nn.Module):

    def __init__(self, in_features, rng):
        super(LinearCell, self).__init__()

        # Sample architecture
        self.out_features = rng.integers(low=15, high=400)
        self.in_features = in_features

        self.linear = torch.nn.Linear(
                in_features=in_features,
                out_features=self.out_features,
            )
        self.drop = torch.nn.Dropout(p=0.5)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.drop(x)
        x = self.act(x)
        return x

