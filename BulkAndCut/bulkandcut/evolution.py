
import os
from copy import deepcopy
from datetime import datetime


import torch

from bulkandcut.model import BNCmodel
from bulkandcut.individual import Individual

class Evolution():
    def __init__(
        self,
        input_shape,
        n_classes:int,
        work_directory:str,
        train_data_loader: "torch.utils.data.DataLoader",
        valid_data_loader: "torch.utils.data.DataLoader",
        initial_population_size:int = 2,
        max_bulk_ups:int = 10,
        max_slim_downs:int = 20,
        cross_entropy_weights:"torch.Tensor" = None,
        ):
        # Just variable initializations
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.work_directory = work_directory
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.initial_population_size = initial_population_size
        self.max_bulk_ups = max_bulk_ups
        self.max_slim_downs = max_slim_downs
        self.population = []
        self.max_num_epochs = 3 #TODO: change to 50  # This is a project constraint

    def __str__(self):
        pass
        #TODO implement


    def _create_work_directory(self):
        if os.path.exists(self.work_directory):
            raise Exception(f"Directory exists: {self.work_directory}")
        os.makedirs(self.work_directory)

    def _get_model_path(self, indv_id:int):
        return os.path.join(
            self.work_directory,
            str(indv_id).rjust(4, "0") + ".pt",
        )

    def _train_initial_population(self):
        for indv_id in range(self.initial_population_size):
            new_model = BNCmodel.NEW(
                input_shape=self.input_shape,
                n_classes=self.n_classes,
                )
            performance = new_model.train_heavylift(
                n_epochs=self.max_num_epochs,
                train_data_loader=self.train_data_loader,
                valid_data_loader=self.valid_data_loader,
                )
            path_to_model = self._get_model_path(indv_id=indv_id)
            new_individual = Individual(
                indv_id=indv_id,
                path_to_model=path_to_model,
                summary=new_model.summary,
                parent_id=None,
                bulk_counter=0,
                cut_counter=0,
                pre_training_loss=performance["pre_training_loss"],
                post_training_loss=performance["post_training_loss"],
                post_training_accuracy=performance["post_training_accuracy"],
                n_parameters=new_model.n_parameters,
                )
            new_model.save(file_path=path_to_model)
            new_individual.save_info()
            self.population.append(new_individual)

    def _slim_down_individual(self, parent_id):
        parent_indv = self.population[parent_id]
        parent_model = BNCmodel.LOAD(parent_indv.path_to_model)
        child_model = parent_model.slimdown()
        performance = child_model.train_cardio(
            n_epochs=3,
            parent_model=parent_model,
            train_data_loader=self.train_data_loader,
            valid_data_loader=self.valid_data_loader,
            )
        child_id = len(self.population)
        path_to_child_model=self._get_model_path(indv_id=child_id)
        new_individual = Individual(
            indv_id=child_id,
            path_to_model=path_to_child_model,
            summary=child_model.summary,
            parent_id=parent_id,
            bulk_counter=parent_indv.bulk_counter,
            cut_counter=parent_indv.cut_counter + 1,
            pre_training_loss=performance["pre_training_loss"],
            post_training_loss=performance["post_training_loss"],
            post_training_accuracy=performance["post_training_accuracy"],
            n_parameters=child_model.n_parameters,
        )
        self.population.append(new_individual)
        child_model.save(file_path=path_to_child_model)
        new_individual.save_info()





    def run(self, time_budget:float):
        run_start = datetime.now()

        self._create_work_directory()
        self._train_initial_population()

        # Check if we still have time:
        while (datetime.now() - run_start).seconds < time_budget:
            self._slim_down_individual(parent_id=1)



            break













