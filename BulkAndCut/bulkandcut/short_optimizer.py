import math
import os
import csv

import numpy as np
import bayes_opt

from bulkandcut import rng

class ShortOptimizer():

    def __init__(self, log_dir:str):
        self.log_path = os.path.join(log_dir, "BO_short.csv")
        parameter_bounds = {
            "lr_exp" : (-5., -2.),
            "w_decay_exp" : (-4., -1.),  # weight_decay = 10^w_decay_exp
        }
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
        neg_training_loss = -learning_curves["train_loss"][-1]
        self.optimizer.register(
            params=config,
            target=neg_training_loss,
        )


        #TODO move to its own function
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

