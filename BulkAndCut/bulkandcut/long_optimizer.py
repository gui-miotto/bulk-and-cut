import math
import os
import csv

import numpy as np
import bayes_opt

from bulkandcut import rng

class LongOptimizer():

    def __init__(self, log_dir:str):
        self.log_path = os.path.join(log_dir, "BO_long.csv")
        parameter_bounds = {
            "lr_exp" : (-4., math.log10(0.05)),  # LR = 10^lr_exp  # TODO: put it back to -1 and test the sequential domain reduction
            "w_decay_exp" : (-4., -1.),  # weight_decay = 10^w_decay_exp
            "lr_sched_gamma" : (.1, 1.),  # 1. is equivalent to no schedule
            "lr_sched_step_size" : (2., 50.),
        }
        # The baseline (default configuration) is included in the search space.
        # default conf = {
        #     "lr_exp" : math.log10(2.244958736283895e-05),
        #     "w_decay_exp" : -2,
        #     "lr_sched_gamma" : 1.,  # No schedule
        #     "lr_sched_step_size" : 25.,  # This is irrelevant, because lr_sched_gamma=1.
        # }
        self.optimizer = bayes_opt.BayesianOptimization(
            f=None,
            pbounds=parameter_bounds,
            verbose=2,
            random_state=1,
        )
        self.utility_func = bayes_opt.UtilityFunction(
            kind="ucb",
            kappa=2.5,
            xi=0.0,
        )


    def next_config(self):
        suggestion = self.optimizer.suggest(utility_function=self.utility_func)
        return suggestion


    def register_results(self, config, learning_curves):
        neg_valid_loss = -learning_curves["validation_loss"][-1]
        self.optimizer.register(
            params=config,
            target=neg_valid_loss,
        )

        # Write configurations and their respective targets on a csv file
        with open(self.log_path, 'w', newline='') as csvfile:
            fieldnames = ["order", "target"] + list(self.optimizer.res[0]["params"].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for n, conf in enumerate(self.optimizer.res):
                row = {
                    "order" : n,
                    "target" : conf["target"],
                    }
                row.update(conf["params"])
                writer.writerow(row)

