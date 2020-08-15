from copy import deepcopy

import numpy as np
import torch
import torchsummary

from bulkandcut.conv_cell import ConvCell
from bulkandcut.linear_cell import LinearCell


class BNCmodel(torch.nn.Module):

    rng = np.random.default_rng(seed=0)

    @classmethod
    def NEW(cls, input_shape, n_classes):
        # Sample
        n_conv_trains = cls.rng.integers(low=1, high=4)

        # Convolutional layers
        conv_trains = torch.nn.ModuleList()
        in_channels = input_shape[0]
        for _ in range(n_conv_trains):
            cc = ConvCell(in_channels=in_channels, rng=cls.rng)
            mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            conv_train = torch.nn.ModuleList([cc, mp])
            conv_trains.append(conv_train)
            in_channels = cc.out_channels

        # Fully connected (i.e. linear) layers
        in_features = cls._get_conv_output(shape=input_shape, conv_trains=conv_trains)
        lc = LinearCell.NEW(in_features=in_features, rng=cls.rng)
        head = torch.nn.Linear(
            in_features=lc.out_features,
            out_features=n_classes,
        )
        linear_train = torch.nn.ModuleList([lc, head])

        return cls(conv_trains=conv_trains, linear_train=linear_train, input_shape=input_shape)


    @classmethod
    def BULKUP(cls, parent):
        conv_trains = deepcopy(parent.conv_trains)
        linear_train = deepcopy(parent.linear_train)

        sel_layer = linear_train[0]
        morph = sel_layer.downstream_morphism()
        linear_train.insert(index=1, module=morph)

        return cls(conv_trains=conv_trains, linear_train=linear_train, input_shape=parent.input_shape)


    @classmethod
    def _get_conv_output(cls, shape, conv_trains):
        bs = 1
        x = torch.autograd.Variable(torch.rand(bs, *shape))
        for ct in conv_trains:
            for module in ct:
                x = module(x)
        n_size = x.data.view(bs, -1).size(1)
        return n_size



    def __init__(self, conv_trains, linear_train, input_shape):
        """
        There is no reason to ever use this constructor directly.
        Instead, use the class methods with ALLCAPS names.
        """

        super(BNCmodel, self).__init__()
        self.conv_trains = conv_trains
        self.linear_train = linear_train
        self.input_shape = input_shape


    def forward(self, x):
        # convolutions and friends
        for ct in self.conv_trains:
            for module in ct:
                x = module(x)
        # flattening
        x = x.view(x.size(0), -1)
        # linear and friends
        for module in self.linear_train:
            x = module(x)
        return x


    def summary(self):
        torchsummary.summary(model=self, input_size=self.input_shape)
