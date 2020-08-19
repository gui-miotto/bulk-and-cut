
import os
import csv
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
        debugging: bool = False,
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
        self.debugging=debugging

        self.population = []
        self.max_num_epochs = 2 #TODO: change to 50  # This is a project constraint

    def save_csv(self):
        file_path = os.path.join(self.work_directory, "population_summary.csv")
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = self.population[0].to_dict().keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for indv in self.population:
                writer.writerow(indv.to_dict())


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
            path_to_model = self._get_model_path(indv_id=indv_id)
            performance = new_model.start_training(
                n_epochs=self.max_num_epochs,
                train_data_loader=self.train_data_loader,
                valid_data_loader=self.valid_data_loader,
                train_fig_path=path_to_model + ".png" if self.debugging else None,
                )
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
        self.save_csv()

    def _generate_offspring(self, parent_id:int, transformation:str):
        if transformation not in ["bulk-up", "slim-down"]:
            raise Exception("Unknown transformation")
        bulking = transformation == "bulk-up"

        parent_indv = self.population[parent_id]
        parent_model = BNCmodel.LOAD(parent_indv.path_to_model)
        child_model = parent_model.bulkup() if bulking else parent_model.slimdown()
        child_id = len(self.population)
        path_to_child_model=self._get_model_path(indv_id=child_id)
        performance = child_model.start_training(
            n_epochs=self.max_num_epochs,
            parent_model=None if bulking else parent_model,
            train_data_loader=self.train_data_loader,
            valid_data_loader=self.valid_data_loader,
            train_fig_path=path_to_child_model + ".png" if self.debugging else None,
            )
        new_individual = Individual(
            indv_id=child_id,
            path_to_model=path_to_child_model,
            summary=child_model.summary,
            parent_id=parent_id,
            bulk_counter=parent_indv.bulk_counter + (1 if bulking else 0),
            cut_counter=parent_indv.cut_counter + (0 if bulking else 1),
            pre_training_loss=performance["pre_training_loss"],
            post_training_loss=performance["post_training_loss"],
            post_training_accuracy=performance["post_training_accuracy"],
            n_parameters=child_model.n_parameters,
        )
        self.population.append(new_individual)
        child_model.save(file_path=path_to_child_model)
        new_individual.save_info()
        self.save_csv()

    def run(self, time_budget:float):
        self._create_work_directory()
        self._train_initial_population()

        self._generate_offspring(parent_id=len(self.population) - 1, transformation="bulk-up")
        self._generate_offspring(parent_id=len(self.population) - 1, transformation="slim-down")


    def run_(self, time_budget:float):
        run_start = datetime.now()

        self._create_work_directory()
        self._train_initial_population()

        # Check if we still have time:
        while (datetime.now() - run_start).seconds < time_budget:
            self._generate_offspring(parent_id=len(self.population) - 1, transformation="bulk-up")
            self._generate_offspring(parent_id=len(self.population) - 1, transformation="slim-down")
            break







