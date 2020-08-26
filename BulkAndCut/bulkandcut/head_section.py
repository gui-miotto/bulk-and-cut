from copy import deepcopy

import torch


class HeadSection(torch.nn.Module):

    @classmethod
    def NEW(cls, in_features, out_features):
        linear_layer = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )
        return HeadSection(linear_layer=linear_layer)

    def __init__(self, linear_layer):
        super(HeadSection, self).__init__()
        self.layer = linear_layer

    @property
    def in_features(self):
        return self.layer.in_features

    @property
    def out_features(self):
        return self.layer.out_features

    def forward(self, x):
        return self.layer(x)

    def bulkup(self):
        return deepcopy(self)

    def slimdown(self, amount:float):
        num_in_features = int((1. - amount) * self.layer.in_features)
        new_layer = torch.nn.Linear(
            in_features=num_in_features,
            out_features=self.out_features,
        )

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.layer.weight),
            dim=0,
        )
        in_selected = torch.argsort(w_l1norm)[-num_in_features:]
        in_selected = torch.sort(in_selected).values  # this is actually not not necessary

        weight = deepcopy(self.layer.weight.data[:,in_selected])
        bias = deepcopy(self.layer.bias)
        new_layer.weight = torch.nn.Parameter(weight)
        new_layer.bias = torch.nn.Parameter(bias)

        narrower_section = HeadSection(linear_layer=new_layer)

        return narrower_section, in_selected
