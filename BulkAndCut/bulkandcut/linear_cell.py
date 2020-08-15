import numpy as np
import torch


class LinearCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_features, rng):
        # Sample
        out_features = rng.integers(low=15, high=400)
        ll = torch.nn.Linear(in_features=in_features, out_features=out_features)
        return cls(linear_layer=ll)

    def __init__(self, linear_layer):
        super(LinearCell, self).__init__()
        # modules
        self.linear = linear_layer
        self.drop = torch.nn.Dropout(p=0.5)
        self.act = torch.nn.ReLU()
        # number of features
        self.in_features = linear_layer.weight.shape[1]
        self.out_features = linear_layer.weight.shape[0]


    def forward(self, x):
        x = self.linear(x)
        x = self.drop(x)
        x = self.act(x)
        return x


    def downstream_morphism(self):
        identity_layer = torch.nn.Linear(
            in_features=self.out_features,
            out_features=self.out_features,
        )

        with torch.no_grad():
            # Initiate weights and biases with the identity function
            torch.nn.init.eye_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-4
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-4

        return LinearCell(linear_layer=identity_layer)
