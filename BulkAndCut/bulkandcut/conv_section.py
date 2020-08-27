from copy import deepcopy

import torch

from bulkandcut.conv_cell import ConvCell
from bulkandcut.skip_connection import SkipConnection

class ConvSection(torch.nn.Module):

    @classmethod
    def NEW(cls, in_channels:int, rng, device):
        first_cell = ConvCell.NEW(
            in_channels=in_channels,
            rng=rng,
            )
        cells = torch.nn.ModuleList([first_cell])
        return ConvSection(cells=cells, rng=rng, device=device)


    def __init__(
        self,
        cells:"torch.nn.ModuleList",
        rng:"numpy.random.Generator",
        device:str,
        skip_cnns:"torch.nn.ModuleList"=None,
        ):
        super(ConvSection, self).__init__()
        self.cells = cells
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.device = device  #TODO: maybe use a package wide definition of device instead of having to pass those around all the time (maybe do this together with the rng)
        self.rng = rng
        self.skip_cnns = torch.nn.ModuleList() if skip_cnns is None else skip_cnns

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
        n_cells = len(self.cells)
        x = self.cells[0](x)
        x_buffer = self._build_forward_buffer(buffer_shape=x.shape)

        for i in range(1, n_cells):
            if i in x_buffer:
                x += x_buffer[i]
            for sk in self.skip_cnns:
                if sk.source == i:
                    x_buffer[sk.destiny] += sk(x)
            x = self.cells[i](x)
        if n_cells + 1 in x_buffer:
            x += x_buffer[n_cells + 1]

        x = self.maxpool(x)
        return x

    def _build_forward_buffer(self, buffer_shape):
        addresses = {skcnn.destiny for skcnn in self.skip_cnns}  # a set
        buffer = {addr : torch.zeros(size=buffer_shape).to(self.device) for addr in addresses}  # a dict
        return buffer


    def bulkup(self):  #TODO: this is exactly the same code used by the linear section. Create parent class?
        # Adds a new cell
        sel_cell = self.rng.integers(low=0, high=len(self.cells))
        identity_cell = self.cells[sel_cell].downstream_morphism()
        new_cell_set = deepcopy(self.cells)
        new_cell_set.insert(index=sel_cell + 1, module=identity_cell)

        # Adjust skip connection addressing
        new_skip_cnns = deepcopy(self.skip_cnns)
        for skcnn in new_skip_cnns:
            skcnn.adjust_addressing(inserted_cell=sel_cell + 1)

        # Stochatically add a skip connection
        if self.rng.random() < 1.6:  # TODO: .6
            candidates = self._skip_connection_candidates()
            if len(candidates) > 0:
                print("\n\nADDED!!!\n\n")  # TODO: delete
                chosen = self.rng.choice(candidates)
                new_skip_connection = SkipConnection(source=chosen[0], destiny=chosen[1])
                new_skip_cnns.append(new_skip_connection)

        deeper_section = ConvSection(cells=new_cell_set, skip_cnns=new_skip_cnns, rng=self.rng, device=self.device)
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



    def slimdown(self, out_selected, amount:float):  #TODO: Again this is exactly the same code used by the linear section. Create parent class?
        narrower_cells = torch.nn.ModuleList()
        for cell in self.cells[::-1]:
            pruned_cell, out_selected = cell.prune(
                out_selected=out_selected,
                amount=amount,
                )
            narrower_cells.append(pruned_cell)
        narrower_section = ConvSection(cells=narrower_cells[::-1], rng=self.rng, device=self.device)
        return narrower_section, out_selected
