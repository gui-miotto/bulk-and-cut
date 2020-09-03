import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from bulkandcut.pareto import _hyper_volume_2D


class TestPareto(unittest.TestCase):

    def test_hypervolume_2D(self):
        ref_point = [1E8, 0.]
        difandre_front = np.array([
            [3.64660000e+04, -8.00280941e+01],
            [4.27571700e+06, -8.13530869e+01],
            ])
        expected_vol = 276.96  # given by the project description
        calc_vol = _hyper_volume_2D(pareto_front=difandre_front, ref_point=ref_point)
        self.assertAlmostEqual(calc_vol, expected_vol, places=2)


if __name__ == '__main__':
    unittest.main()
