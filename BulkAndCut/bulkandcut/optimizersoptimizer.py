import math

import numpy as np
import bayes_opt

class OptimizersOptimizer():

    def __init__(self, loss_type:str, probe_first=list()):
        self.probe_first = probe_first
        self.loss_type = loss_type
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
        if len(self.probe_first) > 0:
            return self.probe_first.pop(0)
        else:
            return self.optimizer.suggest(utility_function=self.utility_func)

        # Baseline (default configuration):
        # suggestion = {
        #     "lr_exp" : math.log10(2.244958736283895e-05),
        #     "w_decay_exp" : -2,
        #     "lr_sched_gamma" : 1.,  # No schedule
        #     "lr_sched_step_size" : 40.,  # This is irrelevant, because lr_sched_gamma=1.
        # }
        #return suggestion


    def register_results(self, config, performance):
        init_loss = performance["initial_loss"]
        final_loss = performance["final_loss"]

        if self.loss_type == "naive":
            target = -final_loss
        elif self.loss_type == "bulkup":
            target = (init_loss - final_loss) / init_loss
        elif self.loss_type == "slimdown":
            target = init_loss - final_loss
        else:
            raise Exception("Invalid loss type")

        self.optimizer.register(
            params=config,
            target=target,
        )


    def top_n_percent(self, n=10.):
        targets = [res["target"] for res in self.optimizer.res]
        n_confs = int(math.ceil(len(targets) * n / 100.))
        selected = np.argsort(targets)[::-1][:n_confs]
        confs = [self.optimizer.res[s]["params"] for s in selected]
        return confs
