from copy import deepcopy

import numpy as np
import torch
import torchsummary

from bulkandcut.conv_cell import ConvCell
from bulkandcut.linear_cell import LinearCell


class BNCmodel(torch.nn.Module):

    rng = np.random.default_rng(seed=1)

    @classmethod
    def NEW(cls, input_shape, n_classes):
        # Sample
        n_conv_trains = cls.rng.integers(low=1, high=4)

        # Convolutional layers
        conv_trains = torch.nn.ModuleList()
        in_channels = input_shape[0]
        for _ in range(n_conv_trains):
            cc = ConvCell.NEW(in_channels=in_channels, rng=cls.rng)
            mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            conv_train = torch.nn.ModuleList([cc, mp])
            conv_trains.append(conv_train)
            in_channels = cc.out_channels

        # Fully connected (i.e. linear) layers
        conv_outputs = cls._get_conv_output(shape=input_shape, conv_trains=conv_trains)
        linear_cell = LinearCell.NEW(in_features=conv_outputs, rng=cls.rng)
        head = torch.nn.Linear(
            in_features=linear_cell.out_features,
            out_features=n_classes,
        )
        linear_train = torch.nn.ModuleList([linear_cell, head])

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=input_shape,
            conv_outputs=conv_outputs,
            )


    @classmethod
    def _get_conv_output(cls, shape, conv_trains):
        bs = 1
        x = torch.autograd.Variable(torch.rand(bs, *shape))
        for ct in conv_trains:
            for module in ct:
                x = module(x)
        n_size = x.data.view(bs, -1).size(1)
        return n_size


    def __init__(self, conv_trains, linear_train, input_shape, conv_outputs):
        super(BNCmodel, self).__init__()
        self.conv_trains = conv_trains
        self.linear_train = linear_train
        self.input_shape = input_shape
        self.conv_outputs = conv_outputs


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


    def bulkup(self):
        conv_trains = deepcopy(self.conv_trains)
        linear_train = deepcopy(self.linear_train)

        if BNCmodel.rng.uniform() < .7:  # There is a p chance of adding a convolutional cell
            sel_train = BNCmodel.rng.integers(
                low=0,
                high=len(conv_trains),
                )
            sel_cell = BNCmodel.rng.integers(
                low=0,
                high=len(conv_trains[sel_train]) - 1,  # Subtract 1 to exclude the maxpooling
                )
            identity_cell = conv_trains[sel_train][sel_cell].downstream_morphism()
            conv_trains[sel_train].insert(
                index=sel_cell + 1,
                module=identity_cell,
                )
        else:  # And a (1-p) chance of adding a linear cell
            sel_cell = BNCmodel.rng.integers(
                low=0,
                high=len(linear_train) - 1,  # Subtract 1 to exclude the head
                )
            identity_cell = linear_train[sel_cell].downstream_morphism()
            linear_train.insert(
                index=sel_cell + 1,
                module=identity_cell,
                )

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=self.input_shape,
            conv_outputs=self.conv_outputs,
            )

    def slimdown(self):
        # self.input_shape[0] is the number of channels in the images
        in_select = list(range(self.input_shape[0]))

        # Prune convolutional cells
        conv_trains = torch.nn.ModuleList()
        for conv_train in self.conv_trains:
            slimmer_conv_train = torch.nn.ModuleList()
            for conv_cell in conv_train[:-1]:  # Subtract 1 to exclude the maxpool
                pruned_conv_cell, in_select = conv_cell.prune(in_select=in_select)
                slimmer_conv_train.append(pruned_conv_cell)
            maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            slimmer_conv_train.append(maxpool)
            conv_trains.append(slimmer_conv_train)

        # Convert filter index to unit index
        in_select = self._flatten_filter_index(in_select)

        # Prune linear cells
        linear_train = torch.nn.ModuleList()
        for linear_cell in self.linear_train[:-1]:  # Subtract 1 to exclude the head
            pruned_linear_cell, in_select = linear_cell.prune(in_select=in_select)
            linear_train.append(pruned_linear_cell)

        # Prune head (just incomming weights, not units, of course)
        head = self._prune_head(in_select=in_select)
        linear_train.append(head)

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=self.input_shape,
            conv_outputs=BNCmodel._get_conv_output(self.input_shape, conv_trains)
            )

    def _prune_head(self, in_select):
        parent_head = self.linear_train[-1]
        n_classes = parent_head.out_features

        head = torch.nn.Linear(
            in_features=len(in_select),
            out_features=n_classes,
        )
        weight = deepcopy(parent_head.weight.data[:,in_select]) # TODO: do I need this deep copy here?
        bias = deepcopy(parent_head.bias) # TODO: do I need this deep copy here?
        head.weight = torch.nn.Parameter(weight)
        head.bias = torch.nn.Parameter(bias)

        return head


    def _flatten_filter_index(self, selection):
        n_filters = self.conv_trains[-1][-2].out_channels
        # sanity check
        if self.conv_outputs % n_filters != 0.0:
            raise Exception("Wrong number of filters")
        upf = self.conv_outputs // n_filters  # units per filter

        # TODO: I'm not quite sure that this corresponds to the way the convolution output is flattten.
        # In other words, am I really selecting the weights of the units I want to keep???
        linear_selection = []
        for s in selection.numpy():
            linear_selection += list(range(s * upf, (s+1) * upf))
        return linear_selection

