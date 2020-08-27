from copy import deepcopy

import torch

from bulkandcut.linear_cell import LinearCell
from bulkandcut.skip_connection import SkipConnection
from bulkandcut import rng, device

class LinearSection(torch.nn.Module):

    @classmethod
    def NEW(cls, in_features):
        first_cell = LinearCell.NEW(in_features=in_features)
        cells = torch.nn.ModuleList([first_cell])
        return LinearSection(cells=cells)

    def __init__(self, cells:"torch.nn.ModuleList", skip_cnns:"torch.nn.ModuleList"=None):
        super(LinearSection, self).__init__()
        self.cells = cells
        self.skip_cnns = torch.nn.ModuleList() if skip_cnns is None else skip_cnns

    def __iter__(self):
        return self.cells.__iter__()

    @property
    def in_features(self):
        return self.cells[0].in_features

    @property
    def out_features(self):
        return self.cells[-1].out_features

    def forward(self, x):
        n_cells = len(self.cells)
        x = self.cells[0](x)
        x_buffer = self._build_forward_buffer(buffer_shape=x.shape)

        for i in range(1, len(self.cells)):
            if i in x_buffer:
                x += x_buffer[i]
            for sk in self.skip_cnns:
                if sk.source == i:
                    x_buffer[sk.destiny] += sk(x)
            x = self.cells[i](x)
        if n_cells + 1 in x_buffer:
            x += x_buffer[n_cells + 1]
        return x

    def _build_forward_buffer(self, buffer_shape):
        addresses = {skcnn.destiny for skcnn in self.skip_cnns}  # a set
        buffer = {addr : torch.zeros(size=buffer_shape).to(device) for addr in addresses}  # a dict
        return buffer

    def bulkup(self):
        # Adds a new cell
        sel_cell = rng.integers(low=0, high=len(self.cells))
        identity_cell = self.cells[sel_cell].downstream_morphism()
        new_cell_set = deepcopy(self.cells)
        new_cell_set.insert(index=sel_cell + 1, module=identity_cell)

        # Adjust skip connection addressing
        new_skip_cnns = deepcopy(self.skip_cnns)
        for skcnn in new_skip_cnns:
            skcnn.adjust_addressing(inserted_cell=sel_cell + 1)

        # Stochatically add a skip connection
        if rng.random() < 1.6:  #TODO .6
            candidates = self._skip_connection_candidates()
            if len(candidates) > 0:
                print("\n\nADDED!!!\n\n")  # TODO: delete
                chosen = rng.choice(candidates)
                new_skip_connection = SkipConnection(source=chosen[0], destiny=chosen[1])
                new_skip_cnns.append(new_skip_connection)

        deeper_section = LinearSection(cells=new_cell_set, skip_cnns=new_skip_cnns)
        return deeper_section

    def _skip_connection_candidates(self):
        if (n_cells := len(self.cells)) < 3:
            return []

        already_connected = [(sk.source, sk.destiny) for sk in self.skip_cnns]
        candidates = []
        for source in range(1, n_cells - 1):
            for destiny in range(source + 2, n_cells + 1):
                if (source, destiny) not in already_connected:
                    candidates.append((source, destiny))

        return candidates


    def slimdown(self, out_selected, amount:float):
        narrower_cells = torch.nn.ModuleList()
        for cell in self.cells[::-1]:
            pruned_cell, out_selected = cell.prune(
                out_selected=out_selected,
                amount=amount,
                )
            narrower_cells.append(pruned_cell)
        narrower_section = LinearSection(cells=narrower_cells[::-1])
        return narrower_section, out_selected