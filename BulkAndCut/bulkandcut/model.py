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
        lc = LinearCell(in_features=in_features, rng=cls.rng)
        head = torch.nn.Linear(
            in_features=lc.out_features,
            out_features=n_classes,
        )
        linear_train = torch.nn.ModuleList([lc, head])

        return cls(conv_trains=conv_trains, linear_train=linear_train, input_shape=input_shape)


    @classmethod
    def BULKUP(cls, parent):
        pass
        # modlist = deepcopy(parent.modlist[-1:])

    #     new_layer = torch.nn.Linear(
    #         in_features=5,
    #         out_features=10,
    #     )


    #     adjusted_last_layer = torch.nn.Linear(
    #         in_features=10,
    #         out_features=parent.modlist[-1].weight.shape[0],
    #     )


    #     with torch.no_grad():
    #         # Initiate weights and biases with the identity function
    #         torch.nn.init.eye_(new_layer.weight)
    #         torch.nn.init.zeros_(new_layer.bias)

    #         # And add some noise to break the simetry
    #         new_layer.weight += torch.rand_like(new_layer.weight) * 1E-4
    #         new_layer.bias += torch.rand_like(new_layer.bias) * 1E-4

            
    #         adjusted_last_layer.weight = torch.nn.Parameter(torch.cat((parent.modlist[-1].weight, parent.modlist[-1].weight), 1) / 2)
            
      
    #     modlist.append(new_layer)
    #     modlist.append(adjusted_last_layer)

    #     # modlist.insert(
    #     #     index=1,
    #     #     module=new_layer,
    #     #     )


    #     return cls(
    #         input_shape=parent.input_shape,
    #         modlist=modlist,
    #     )

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
