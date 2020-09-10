import unittest
from datetime import datetime

import numpy as np
import torch

import bulkandcut as bnc


def _super_easy_dataset(n_samples: int, n_classes: int = 10, height: int = 8, width: int = 8):
    # random labels
    rng = np.random.default_rng(seed=0)
    labels = rng.integers(low=0, high=n_classes, size=n_samples)
    targets = torch.LongTensor(labels)

    # data is zero for all channels, except the one corresponding to its label
    data = np.zeros((n_samples, n_classes, height, width))
    data[range(n_samples), labels] = 1.
    data = torch.Tensor(data)

    # create dataset
    dataset = torch.utils.data.TensorDataset(data, targets)
    return dataset


class TestEvolution(unittest.TestCase):

    def test_run(self):
        n_classes = 10
        width, height = 8, 8
        # Training data
        train_ds = _super_easy_dataset(
            n_samples=1000,
            n_classes=n_classes,
            height=height,
            width=width,
            )
        train_dl = torch.utils.data.DataLoader(
            dataset=train_ds,
            shuffle=True,
            batch_size=100,
            )
        # Validation data
        valid_ds = _super_easy_dataset(
            n_samples=100,
            n_classes=n_classes,
            height=height,
            width=width,
            )
        valid_dl = torch.utils.data.DataLoader(
            dataset=valid_ds,
            shuffle=False,
            batch_size=100,
            )

        # Run evolution twice: with and without debugging
        for i in range(2):
            evo = bnc.Evolution(
                input_shape=(n_classes, width, height),  # for this dataset n_channels == n_classes
                n_classes=n_classes,
                work_directory=f"/tmp/bulk_and_cut_test/{str(datetime.now())}/",
                train_data_loader=train_dl,
                valid_data_loader=valid_dl,
                debugging=i == 0,
            )
            evo.run(time_budget=60.)  # run it for one minute

            # Make sure the system is learning
            max_accuracy = max([ind.post_training_accuracy for ind in evo.population])
            self.assertGreater(max_accuracy, 95.)


if __name__ == '__main__':
    unittest.main()
