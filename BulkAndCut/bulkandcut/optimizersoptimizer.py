import math

import numpy as np
import bayes_opt

from bulkandcut import rng

class OptimizersOptimizer():

    def __init__(self, loss_type:str, probe_first=list()):
        self.probe_first = probe_first
        self.loss_type = loss_type
        parameter_bounds = {
            "lr_exp" : (-5., -2.),  # LR = 10^lr_exp
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
        if len(self.probe_first) > 0:
            return self.probe_first.pop(0)
        else:
            return self.optimizer.suggest(utility_function=self.utility_func)




    def register_results(self, config, learning_curves):
        init_loss = learning_curves["validation_loss"][0]
        final_loss = learning_curves["validation_loss"][-1]

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


    def sample_n_from_top_p_percent(self, n:int = 20, p:float = 25.):
        targets = [res["target"] for res in self.optimizer.res]
        cutoff = int(math.ceil(len(targets) * p / 100.))
        top_p = np.argsort(targets)[::-1][:cutoff]
        if len(top_p) > n:
            selected = rng.choice(top_p[1:], size=n - 1, replace=False)
            selected = np.hstack((top_p[0], selected))  # always include the very best
        else:
            selected = top_p
        confs = [self.optimizer.res[s]["params"] for s in selected]
        return confs
