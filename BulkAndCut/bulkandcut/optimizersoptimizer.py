import math

import bayes_opt

class OptimizersOptimizer():

    def __init__(self):
        parameter_bounds = {
            "lr_exp" : (-5., -2.),  # LR = 10^lr_exp
            "w_decay_exp" : (-3., -1.),  # weight_decay = 10^w_decay_exp
            "lr_sched_gamma" : (.1, 1.),  # 1. is equivalent to no schedule
            "lr_sched_step_size" : (10., 40.),
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
        suggestion = self.optimizer.suggest(
            utility_function=self.utility_func,
            )
        # Baseline (default configuration):
        # suggestion = {
        #     "lr_exp" : math.log10(2.244958736283895e-05),
        #     "w_decay_exp" : -2,
        #     "lr_sched_gamma" : 1.,  # No schedule
        #     "lr_sched_step_size" : 40.,  # This is irrelevant, because lr_sched_gamma=1.
        # }
        return suggestion


    def register_results(self, config, valid_loss):
        self.optimizer.register(
            params=config,
            target=-valid_loss,
        )
