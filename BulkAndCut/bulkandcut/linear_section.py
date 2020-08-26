from copy import deepcopy

import torch

from bulkandcut.linear_cell import LinearCell


class LinearSection(torch.nn.Module):

    @classmethod
    def NEW(cls, in_features, rng):
        first_cell = LinearCell.NEW(
            in_features=in_features,
            rng=rng,
            )
        cells = torch.nn.ModuleList([first_cell])
        return LinearSection(cells=cells, rng=rng)

    def __init__(self, cells:"torch.nn.ModuleList", rng:"numpy.random.Generator"):
        super(LinearSection, self).__init__()
        self.cells = cells
        self.rng = rng

    def __iter__(self):
        return self.cells.__iter__()

    @property
    def in_features(self):
        return self.cells[0].in_features

    @property
    def out_features(self):
        return self.cells[-1].out_features

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        return x


    def bulkup(self):
        sel_cell = self.rng.integers(low=0, high=len(self.cells))
        identity_cell = self.cells[sel_cell].downstream_morphism()
        new_cell_set = deepcopy(self.cells)
        new_cell_set.insert(index=sel_cell + 1, module=identity_cell)
        deeper_section = LinearSection(cells=new_cell_set, rng=self.rng)
        return deeper_section


    def slimdown(self, out_selected, amount:float):
        narrower_cells = torch.nn.ModuleList()
        for cell in self.cells[::-1]:
            pruned_cell, out_selected = cell.prune(
                out_selected=out_selected,
                amount=amount,
                )
            narrower_cells.append(pruned_cell)
        narrower_section = LinearSection(cells=narrower_cells[::-1], rng=self.rng)
        return narrower_section, out_selected
