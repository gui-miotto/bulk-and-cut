from collections import namedtuple
from copy import deepcopy

import torch

from bulkandcut.model import BNCmodel



Individual = namedtuple(
    typename="Individual",
    field_names=[
        "id",
        "parent_id",
        "bulk_level",
        "cut_level",
        "pre_training_loss",
        "post_training_loss",
        "n_parameters",
        "model",
        ],
    )

class Evolution():
    def __init__(
        self,
        input_shape,
        n_classes:int ,
        initial_population_size:int = 10,
        max_bulk_ups:int = 10,
        max_slim_downs:int = 20,
        cross_entropy_weights:"Optional[torch.Tensor]" = None,
        ):
        # Just copying arguments:
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.initial_population_size = initial_population_size
        self.max_bulk_ups = max_bulk_ups
        self.max_slim_downs = max_slim_downs
        #self.original_population = []
        self.population = []
        # Actually doing something:
        
        #self._initiate_population()
        


    def _initiate_population(self):
        for ind_id in range(self.initial_population_size):
            new_model = BNCmodel.NEW(
                input_shape=self.input_shape,
                n_classes=self.n_classes,
                )
            new_individual = Individual(
                id=ind_id,
                parent_id=None,
                bulk_level=0,
                cut_level=0,
                model=new_model,
                n_parameters=new_model.n_parameters,
                pre_training_loss=None,  # TODO: run a eval to get this loss here?
                post_training_loss=None,
            )
            #self.original_population.append(new_individual)
            self.population.append(new_individual)



    # def _reset_population(self):
    #     self.population = deepcopy(self.original_population)



    def run(self, train_data_loader, valid_data_loader):
        self._initiate_population()

        self.population[0].model.train_heavylift(n_epochs=50, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader)











