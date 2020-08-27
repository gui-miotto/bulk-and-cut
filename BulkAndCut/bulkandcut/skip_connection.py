

import torch

class SkipConnection(torch.nn.Module):

    def __init__(self, source:int, destiny:int):
        super(SkipConnection, self).__init__()
        initial_gain = torch.rand(1) * 1E-6
        self.gain = torch.nn.Parameter(data=initial_gain, requires_grad=True)

        self.source = source
        self.destiny = destiny

    def forward(self, x):
        x = self.gain * x
        return x

    def adjust_addressing(self, inserted_cell:int):
        if self.source > inserted_cell:
            self.source += 1
        if self.destiny > inserted_cell:
            self.destiny += 1


