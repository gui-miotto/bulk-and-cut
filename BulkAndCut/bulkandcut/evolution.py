
import os
import csv
from copy import deepcopy
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from bulkandcut.model import BNCmodel
from bulkandcut.blind_model import BlindModel
from bulkandcut.individual import Individual
from bulkandcut.short_optimizer import ShortOptimizer
from bulkandcut.long_optimizer import LongOptimizer
from bulkandcut import rng, device

class Evolution():

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

        self.short_optimizer = ShortOptimizer(log_dir=work_directory)
        self.long_optimizer = LongOptimizer(log_dir=work_directory)

    @property
    def pop_size(self):
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

    def _train_blind_individual(self, super_stupid:bool):
        indv_id = self.pop_size
        new_model = BlindModel(n_classes=self.n_classes, super_stupid=super_stupid).to(device)
        path_to_model = self._get_model_path(indv_id=indv_id)
        print("Training model", indv_id, "(blind model)")
        learning_curves = new_model.start_training(
            train_data_loader=self.train_data_loader,
            valid_data_loader=self.valid_data_loader,
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
            optimizer_config={},
            learning_curves=learning_curves,
            n_parameters=new_model.n_parameters,
            )
        new_model.save(file_path=path_to_model)
        new_individual.save_info()
        self.population.append(new_individual)
        self.save_csv()


    def _train_naive_individual(self):
        indv_id = self.pop_size
        optm_config = self.long_optimizer.next_config()
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
        self.long_optimizer.register_results(
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


    def _train_offspring(self, parent_id:int, transformation:str):
        if transformation not in ["bulk-up", "slim-down"]:
            raise Exception("Unknown transformation")
        bulking = transformation == "bulk-up"

        parent_indv = self.population[parent_id]
        parent_indv.bulk_offsprings += (1 if bulking else 0)
        parent_indv.cut_offsprings += (0 if bulking else 1)
        parent_model = BNCmodel.LOAD(parent_indv.path_to_model)

        opt_config = parent_indv.optimizer_config if bulking else self.short_optimizer.next_config()
        child_model = parent_model.bulkup(optim_config=opt_config) if bulking else \
                      parent_model.slimdown(optim_config=opt_config)
        child_id = self.pop_size
        path_to_child_model=self._get_model_path(indv_id=child_id)
        print("Training model", child_id)
        learning_curves = child_model.start_training(
            n_epochs=self.max_num_epochs if bulking else int(self.max_num_epochs / 3.),
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
        if not bulking:
            self.short_optimizer.register_results(
                config=opt_config,
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
            optimizer_config=opt_config,
            learning_curves=learning_curves,
            n_parameters=child_model.n_parameters,
        )
        self.population.append(new_individual)
        child_model.save(file_path=path_to_child_model)
        new_individual.save_info()
        self.save_csv()


    def _select_individual_to_reproduce(self, transformation:str):
        if transformation not in ["bulk-up", "slim-down"]:
            raise Exception("Unknown transformation")

        # Selection using the "Paretslon-greedy" method, a combination of epslon-greedy
        # and non-dominated sorting. With a probability epslon, it selects a random
        # individual from the Pareto front. with probability (1 - epslon) it selects a
        # random individual from the 2nd Pareto front, as determined by the non-dominated
        # sorting method.
        pareto_fronts = self._non_dominated_sorting(n_fronts=2)
        front_number = 0 if rng.random() < .85 or len(pareto_fronts[1]) == 0 else 1
        candidates = set(pareto_fronts[front_number])

        # Deal with some exclusions:
        # First, blind models are sterile. :-)
        candidates -= set([0, 1])
        # Then exclude others depending on the transformation
        if transformation == "bulk-up":
            # Exclude individuals that are already too big:
            candidates -= set([i.indv_id for i in self.population if i.n_parameters > int(1E8)])
            # Lets give more probability of selection to models with high accuracy
            candidates = list(candidates)  # back to an ordered data structure
            accuracies = [self.population[ind_id].post_training_accuracy for ind_id in candidates]
            accuracies = np.array(accuracies) / np.sum(accuracies)  # make it sum to 1
            chosen = rng.choice(candidates, p=accuracies)
        else:
            # Exclude individuals that are already too small:
            candidates -= set([i.indv_id for i in self.population if i.n_parameters < int(1E2)])
            # If possible, exclude individuals that have already been slimed-down:
            already_cut = set([i.indv_id for i in self.population if i.cut_offsprings > 0])
            if len(candidates - already_cut) > 0:
                candidates -= already_cut
            chosen = rng.choice(list(candidates))

        return chosen


    def _get_pareto_front(self, exclude_list=[]):
        # TODO: This function is not perfect: In the rare case of where two identical
        # solutions occur and they are not dominated, none of them will be put in the front.
        # Fix this.
        indv_id, num_of_pars, neg_accuracy,  = [], [], []
        for indv in self.population:
            if indv.indv_id not in exclude_list:
                num_of_pars.append(indv.n_parameters)
                neg_accuracy.append(-indv.post_training_accuracy)
                indv_id.append(indv.indv_id)
        if (n_indiv := len(indv_id)) == 0:
            return []
        num_of_pars = np.array(num_of_pars)[:,np.newaxis]
        neg_accuracy = np.array(neg_accuracy)[:,np.newaxis]
        not_eye = np.logical_not(np.eye(n_indiv))  # False in the main diag, True elsew.
        indv_id = np.array(indv_id)

        worst_at_num_pars = np.less_equal(num_of_pars, num_of_pars.T)
        worst_at_accuracy = np.less_equal(neg_accuracy, neg_accuracy.T)
        worst_at_both = np.logical_and(worst_at_num_pars, worst_at_accuracy)
        worst_at_both = np.logical_and(worst_at_both, not_eye)  # excludes self-comparisons
        domination = np.any(worst_at_both, axis=0)

        pareto_front = indv_id[np.logical_not(domination)]
        return list(pareto_front)

    def _non_dominated_sorting(self, n_fronts:int):
        pareto_fronts = []  # This will become a list of lists
        exclude_list = []
        for _ in range(n_fronts):
            front = self._get_pareto_front(exclude_list=exclude_list)
            pareto_fronts.append(front)
            exclude_list.extend(front)
        return pareto_fronts


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


    def run(self, time_budget:float=None, runs_budget:int=None, budget_split:list = [.25, .35, .40]):
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
        init_pop_begin = datetime.now()
        self._train_blind_individual(super_stupid=True)
        self._train_blind_individual(super_stupid=False)
        while True:
            remaining = init_pop_budget - (datetime.now() - init_pop_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the initial phase")
            self._train_naive_individual()

        #Phase 2: Bulk-up  # TODO: two times the same code (phase 2 and phase 3). Merge?
        print("Starting phase 2: Bulk-up")
        bulkup_budget = budget_split[1] * time_budget
        bulkup_begin = datetime.now()
        while True:
            remaining = bulkup_budget - (datetime.now() - bulkup_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the bulk-up phase")
            to_bulk = self._select_individual_to_reproduce(transformation="bulk-up")
            self._train_offspring(parent_id=to_bulk, transformation="bulk-up")

        # Phase 3: Slim-down
        print("Starting phase 3: Slim-down")
        slimdown_budget = budget_split[2] * time_budget
        slimdown_begin = datetime.now()
        while True:
            remaining = slimdown_budget - (datetime.now() - slimdown_begin).seconds
            if remaining < 0:
                break
            print(f"Still {remaining / 60.:.1f} minutes left for the slim-down phase")
            to_cut = self._select_individual_to_reproduce(transformation="slim-down")
            self._train_offspring(parent_id=to_cut, transformation="slim-down")


    #TODO: delete
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


    #TODO: delete
    def run_2(self, time_budget:float=None, runs_budget:int=None, budget_split:list = [.3, .3, .4]):
        if (time_budget is None) == (runs_budget is None):
            raise Exception("One (and only one) of the bugets has to be informed")
        if runs_budget is not None:
            raise Exception("Not implemented yet")  #TODO: implement
        if len(budget_split) != 3 or np.sum(budget_split) != 1.:
            raise Exception("Bad budget split")

        #Phase 2: Bulk-up
        print("Starting phase 2: Bulk-up")
        bulkup_budget = budget_split[1] * time_budget
        bulkup_start_time = datetime.now()
        bulk_level_pointer = 0
        while (datetime.now() - bulkup_start_time).seconds < bulkup_budget:
            to_bulk = self._select_individual_to_reproduce(transformation="bulk-up")
            if to_bulk is not None:
                self._train_offspring(parent_id=to_bulk, transformation="bulk-up")
            bulk_level_pointer = (bulk_level_pointer + 1) % self.max_bulk_ups

        #Optimizer's optimizer knowlegde transfer:
        opt_bulkup_top_confs = self.optm_optm_bulkup.top_n_percent()
        #self.optm_optm_slimdown.probe_first = self.opt_naive_top_confs + opt_bulkup_top_confs  #TODO: test this when unifiying the functions again
        self.optm_optm_slimdown.probe_first = list(self.optm_optm_bulkup.probe_first)

        # Phase 3: Slim-down
        print("Starting phase 3: Slim-down")
        slimdown_budget = budget_split[2] * time_budget
        slimdown_start_time = datetime.now()
        while (datetime.now() - slimdown_start_time).seconds < slimdown_budget:
            to_cut = self._select_individual_to_reproduce(transformation="slim-down")
            self._train_offspring(parent_id=to_cut, transformation="slim-down")
