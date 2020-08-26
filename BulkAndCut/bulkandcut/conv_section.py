from copy import deepcopy

import torch

from bulkandcut.conv_cell import ConvCell


class ConvSection(torch.nn.Module):

    @classmethod
    def NEW(cls, in_channels:int, rng):
        first_cell = ConvCell.NEW(
            in_channels=in_channels,
            rng=rng,
            )
        cells = torch.nn.ModuleList([first_cell])
        return ConvSection(cells=cells, rng=rng)


    def __init__(self, cells:"torch.nn.ModuleList", rng:"numpy.random.Generator"):
        super(ConvSection, self).__init__()
        self.cells = cells
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.rng = rng

    def mark_as_first_section(self):
        self.cells[0].is_first_cell = True

    def __iter__(self):
        return self.cells.__iter__()

    @property
    def in_channels(self):
        return self.cells[0].in_channels

    @property
    def out_channels(self):
        return self.cells[-1].out_channels

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        x = self.maxpool(x)
        return x


    def bulkup(self):  #TODO: this is exactly the same code used by the linear section. Create parent class?
        sel_cell = self.rng.integers(low=0, high=len(self.cells))
        identity_cell = self.cells[sel_cell].downstream_morphism()
        new_cell_set = deepcopy(self.cells)
        new_cell_set.insert(index=sel_cell + 1, module=identity_cell)
        deeper_section = ConvSection(cells=new_cell_set, rng=self.rng)
        return deeper_section


    def slimdown(self, out_selected, amount:float):  #TODO: Again this is exactly the same code used by the linear section. Create parent class?
        narrower_cells = torch.nn.ModuleList()
        for cell in self.cells[::-1]:
            pruned_cell, out_selected = cell.prune(
                out_selected=out_selected,
                amount=amount,
                )
            narrower_cells.append(pruned_cell)
        narrower_section = ConvSection(cells=narrower_cells[::-1], rng=self.rng)
        return narrower_section, out_selected
