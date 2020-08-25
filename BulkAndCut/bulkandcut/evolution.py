
import os
import csv
from copy import deepcopy
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from bulkandcut.model import BNCmodel
from bulkandcut.individual import Individual
from bulkandcut.optimizersoptimizer import OptimizersOptimizer

class Evolution():

    rng = np.random.default_rng(seed=1)  #TODO: this should come from above, so that we seed the whole thing (torch, numpy, cross-validation splits just at one place)

    def __init__(
        self,
        input_shape,
        n_classes:int,
        work_directory:str,
        train_data_loader: "torch.utils.data.DataLoader",
        valid_data_loader: "torch.utils.data.DataLoader",
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
        self.max_bulk_ups = max_bulk_ups
        self.max_slim_downs = max_slim_downs
        self.max_bulk_offsprings_per_individual = max_bulk_offsprings_per_individual
        self.debugging=debugging

        self.population = []
        self.max_num_epochs = 50  #minimum two

        self.optm_optm_naive = OptimizersOptimizer(loss_type="naive")
        self.optm_optm_bulkup = OptimizersOptimizer(loss_type="bulkup")
        self.optm_optm_slimdown = OptimizersOptimizer(loss_type="slimdown")

    @property
    def pop_size(self):
        """
        Population size
        """
        return len(self.population)

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


    def _train_naive_individual(self):
        indv_id = self.pop_size
        optm_config = self.optm_optm_naive.next_config()
        new_model = BNCmodel.NEW(
            input_shape=self.input_shape,
            n_classes=self.n_classes,
            optimizer_configuration=optm_config,
            )
        path_to_model = self._get_model_path(indv_id=indv_id)
        print("Training model", indv_id)
        learning_curves = new_model.start_training(
            n_epochs=self.max_num_epochs,
            train_data_loader=self.train_data_loader,
            valid_data_loader=self.valid_data_loader,
            return_all_learning_curvers=self.debugging,
            )
        self._plot_learning_curves(
            fig_path=path_to_model + ".png",
            curves=learning_curves,
        )
        self.optm_optm_naive.register_results(
            config=optm_config,
            learning_curves=learning_curves,
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
            optimizer_config=optm_config,
            learning_curves=learning_curves,
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

        optim_optim = self.optm_optm_bulkup if bulking else self.optm_optm_slimdown
        optm_config = optim_optim.next_config()
        child_model = parent_model.bulkup(optim_config=optm_config) if bulking else \
                      parent_model.slimdown(optim_config=optm_config)
        child_id = self.pop_size
        path_to_child_model=self._get_model_path(indv_id=child_id)
        print("Training model", child_id)
        learning_curves = child_model.start_training(
            n_epochs=self.max_num_epochs if bulking else int(self.max_num_epochs / 2),  #TODO: tune
            teacher_model=None if bulking else parent_model,
            train_data_loader=self.train_data_loader,
            valid_data_loader=self.valid_data_loader,
            return_all_learning_curvers=self.debugging,
            )
        self._plot_learning_curves(
            fig_path=path_to_child_model + ".png",
            curves=learning_curves,
            parent_loss=parent_indv.post_training_loss,
            parent_accuracy=parent_indv.post_training_accuracy,
        )
        optim_optim.register_results(
            config=optm_config,
            learning_curves=learning_curves,
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
            optimizer_config=optm_config,
            learning_curves=learning_curves,
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
               indv.n_parameters > int(1E8) or \
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
        candidates = []
        for indv in self.population:
            # Exclusion criteria:
            #if indv.cut_offsprings != 0 or \
            if indv.indv_id not in pareto_front or \
               indv.n_parameters < int(1E4):
                continue
            candidates.append(indv.indv_id)

        if len(candidates) == 0:
            print("Warning: No candidates to slim down!")  #TODO: use log instead?
            return None

        return Evolution.rng.choice(candidates)


    def _get_pareto_front(self):
        # TODO: this function is not perfect. Compare with the one in the pareto.py (Unify?!)
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

    def _plot_learning_curves(self, fig_path, curves, parent_loss=None, parent_accuracy=None):
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(curves["validation_loss"], label="valid", color=color)
        ax1.plot(curves["train_loss"], label="train", color="tab:orange")
        ax1.plot(curves["train_loss_at_eval"], label="valid", color="tab:pink")
        if parent_loss is not None:
            ax1.axhline(parent_loss, color=color, linestyle="--")
        ax1.tick_params(axis='y', labelcolor=color)
        #plt.legend([tloss, vloss], ['train','valid'])  # TODO: legend not working

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(curves["validation_accuracy"], color=color)
        ax2.plot(curves["train_accuracy"], color="b")
        if parent_accuracy is not None:
            ax2.axhline(parent_accuracy, color=color, linestyle="--")
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #plt.legend()
        plt.savefig(fig_path)
        plt.close()


    def run_1(self, time_budget:float=None, runs_budget:int=None, budget_split:list = [.3, .3, .4]):
        if (time_budget is None) == (runs_budget is None):
            raise Exception("One (and only one) of the bugets has to be informed")
        if runs_budget is not None:
            raise Exception("Not implemented yet")  #TODO: implement
        if len(budget_split) != 3 or np.sum(budget_split) != 1.:
            raise Exception("Bad budget split")

        self._create_work_directory()

        # Phase 1: Initiated population
        print("Starting phase 1: Initiate population")
        init_pop_budget = budget_split[0] * time_budget
        init_pop_start_time = datetime.now()
        while (datetime.now() - init_pop_start_time).seconds < init_pop_budget:
            self._train_naive_individual()

        # Optimizer's optimizer knowlegde transfer:
        opt_naive_top_confs = self.optm_optm_naive.top_n_percent()
        self.optm_optm_bulkup.probe_first = opt_naive_top_confs


    def run_2(self, time_budget:float=None, runs_budget:int=None, budget_split:list = [.3, .3, .4]):
        if (time_budget is None) == (runs_budget is None):
            raise Exception("One (and only one) of the bugets has to be informed")
        if runs_budget is not None:
            raise Exception("Not implemented yet")  #TODO: implement
        if len(budget_split) != 3 or np.sum(budget_split) != 1.:
            raise Exception("Bad budget split")

        # Phase 2: Bulk-up
        # print("Starting phase 2: Bulk-up")
        # bulkup_budget = budget_split[1] * time_budget
        # bulkup_start_time = datetime.now()
        # bulk_level_pointer = 0
        # while (datetime.now() - bulkup_start_time).seconds < bulkup_budget:
        #     to_bulk = self._select_individual_to_bulkup(bulk_level=bulk_level_pointer)
        #     if to_bulk is not None:
        #         self._generate_offspring(parent_id=to_bulk, transformation="bulk-up")
        #     bulk_level_pointer = (bulk_level_pointer + 1) % self.max_bulk_ups

        # Optimizer's optimizer knowlegde transfer:
        #opt_bulkup_top_confs = self.optm_optm_bulkup.top_n_percent()
        #self.optm_optm_slimdown.probe_first = self.opt_naive_top_confs + opt_bulkup_top_confs  #TODO: test this when unifiying the functions again
        self.optm_optm_slimdown.probe_first = list(self.optm_optm_bulkup.probe_first)

        # Phase 3: Slim-down
        print("Starting phase 3: Slim-down")
        slimdown_budget = budget_split[2] * time_budget
        slimdown_start_time = datetime.now()
        while (datetime.now() - slimdown_start_time).seconds < slimdown_budget:
            if (to_cut := self._select_individual_to_slimdown()) is None:
                break
            self._generate_offspring(parent_id=to_cut, transformation="slim-down")



    def debug_fn(self, time_budget:float):
        # Just for debugging #TODO: delete
        self._create_work_directory()
        self._train_initial_population()

        self._generate_offspring(parent_id=len(self.population) - 1, transformation="bulk-up")
        self._generate_offspring(parent_id=len(self.population) - 1, transformation="slim-down")
