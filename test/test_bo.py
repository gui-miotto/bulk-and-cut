import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from bulkandcut.bayesian_optimization.constrained_bayesian_optimizer \
    import ConstrainedBayesianOptimizer as CBO


def sphere(x1, x2, x1min=0., x2min=0.):
    y = np.square(x1 - x1min) + np.square(x2 - x2min)
    return y


class TestBO(unittest.TestCase):

    def test_minimize_sphere(self):
        xmin = [.5, -.35]
        pbounds = {
            "x1" : (-1., 1.),
            "x2" : (-1., 1.),
        }

        cbo = CBO(par_bounds=pbounds)

        n_iters = 50
        for n in range(n_iters):
            print(f"iteration {n}/{n_iters}\r", end="")
            pars = cbo.next_pars(dictated_pars={})
            target = sphere(pars["x1"], pars["x2"], xmin[0], xmin[1])
            cbo.register_target(par_values=pars, target=target)
        print("Argmin", cbo.incumbent)

        assert_almost_equal(actual=cbo.incumbent, desired=xmin, decimal=2)


    def test_minimize_sphere_constrained(self):
        pbounds = {
            "x1" : (-1., 1.),
            "x2" : (-1., 1.),
        }

        cbo = CBO(par_bounds=pbounds)
        x2_constrain = .6
        xmin = [0., x2_constrain]

        n_iters = 50
        for n in range(n_iters):
            print(f"iteration {n}/{n_iters}\r", end="")
            pars = cbo.next_pars(dictated_pars={"x2" : x2_constrain})
            target = sphere(pars["x1"], pars["x2"])
            cbo.register_target(par_values=pars, target=target)
        print("Argmin", cbo.incumbent)

        assert_almost_equal(actual=cbo.incumbent, desired=xmin, decimal=2)


if __name__ == '__main__':
    unittest.main()