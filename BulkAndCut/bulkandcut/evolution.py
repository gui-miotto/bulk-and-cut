
import os
import csv
from copy import deepcopy
from datetime import datetime

import torch
import numpy as np

from bulkandcut.model import BNCmodel
from bulkandcut.individual import Individual

class Evolution():

    rng = np.random.default_rng(seed=1)  #TODO: this should come from above, so that we seed the whole thing (torch, numpy, cross-validation splits just at one place)

    def __init__(
        self,
        input_shape,
        n_classes:int,
        work_directory:str,
        train_data_loader: "torch.utils.data.DataLoader",
        valid_data_loader: "torch.utils.data.DataLoader",
        initial_population_size:int = 100,
        max_bulk_ups:int = 6,
        max_slim_downs:int = 20,
        max_bulk_offsprings_per_individual:int = 2,
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
        self.max_bulk_offsprings_per_individual = max_bulk_offsprings_per_individual
        self.debugging=debugging

        self.population = []
        self.max_num_epochs = 50

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
            print("Training model", indv_id)
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
                parent_id=-1,  # No parent
                bulk_counter=0,
                cut_counter=0,
                bulk_offsprings=0,
                cut_offsprings=0,
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
        parent_indv.bulk_offsprings += (1 if bulking else 0)
        parent_indv.cut_offsprings += (0 if bulking else 1)
        parent_model = BNCmodel.LOAD(parent_indv.path_to_model)

        child_model = parent_model.bulkup() if bulking else parent_model.slimdown_()
        child_id = len(self.population)
        path_to_child_model=self._get_model_path(indv_id=child_id)
        print("Training model", child_id)
        performance = child_model.start_training(
            n_epochs=self.max_num_epochs if bulking else int(self.max_num_epochs / 2),
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
            bulk_offsprings=0,
            cut_offsprings=0,
            pre_training_loss=performance["pre_training_loss"],
            post_training_loss=performance["post_training_loss"],
            post_training_accuracy=performance["post_training_accuracy"],
            n_parameters=child_model.n_parameters,
        )
        self.population.append(new_individual)
        child_model.save(file_path=path_to_child_model)
        new_individual.save_info()
        self.save_csv()


    def _select_individual_to_bulkup(self, bulk_level:int):
        # Generate list of suitable candidates and store their losses
        candidates, losses = [], []
        for indv in self.population:
            # Exclusion criteria:
            if indv.bulk_counter != bulk_level or \
               indv.cut_counter > 0 or \
               indv.bulk_offsprings >= self.max_bulk_offsprings_per_individual:
                continue

            candidates.append(indv.indv_id)
            losses.append(indv.post_training_loss)

        # No suitable candidates
        if len(candidates) == 0:
            print("Warning: No candidates to bulk up!")  #TODO: use log instead?
            return None

        # Epslon-greedy selection:
        if Evolution.rng.random() < .8:
            # return the candidate with the smallest loss
            return candidates[np.argmin(losses)]
        else:
            # return a random candidate
            return Evolution.rng.choice(candidates)


    def _select_individual_to_slimdown(self):
        # Returns a random individual in the pareto front, that has never slimmed down
        pareto_front = self._get_pareto_front()
        candidates = [iid for iid in pareto_front if self.population[iid].cut_offsprings == 0]

        if len(candidates) == 0:
            print("Warning: No candidates to slim down!")  #TODO: use log instead?
            return None

        return Evolution.rng.choice(candidates)


    def _get_pareto_front(self):
        n_pars = np.array([indv.n_parameters for indv in self.population])
        accs = np.array([indv.post_training_accuracy for indv in self.population])

        pareto_front = []
        for indv in self.population:
            npars_comp = n_pars < indv.n_parameters
            accs_comp = accs > indv.post_training_accuracy
            domination = np.logical_and(npars_comp, accs_comp)
            if not np.any(domination):
                pareto_front.append(indv.indv_id)

        return pareto_front


    def run(self, time_budget:float):
        run_start = datetime.now()

        self._create_work_directory()
        self._train_initial_population()
        bulk_level_pointer = 0

        # Check if we still have time:
        while (datetime.now() - run_start).seconds < time_budget:
            # Bulk a model up:
            to_bulk = self._select_individual_to_bulkup(bulk_level=bulk_level_pointer)
            bulk_level_pointer = (bulk_level_pointer + 1) % self.max_bulk_ups
            if to_bulk is not None:
                self._generate_offspring(parent_id=to_bulk, transformation="bulk-up")

            # Slim two models down:
            for _ in range(2):
                to_cut = self._select_individual_to_slimdown()
                if to_cut is not None:
                    self._generate_offspring(parent_id=to_cut, transformation="slim-down")


    def debug_fn(self, time_budget:float):
        # Just for debugging
        self._create_work_directory()
        self._train_initial_population()

        self._generate_offspring(parent_id=len(self.population) - 1, transformation="bulk-up")
        self._generate_offspring(parent_id=len(self.population) - 1, transformation="slim-down")
