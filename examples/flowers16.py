import os
from datetime import datetime

import torch
import sklearn.model_selection
import numpy as np

import bulkandcut as bnc


# Provided benchmarks:
ref_point = [1E8, 0.]
benchmarks = [
    bnc.Benchmark(
        name="baseline",
        plot_front=True,
        marker="o",
        color="tab:green",
        data=np.array([
            [2.84320000e+04, -5.86384692e+01],
            [8.80949400e+06, -7.69414740e+01],
            ])),
    bnc.Benchmark(
        name="difandre",
        plot_front=True,
        marker="o",
        color="tab:blue",
        data=np.array([
            [3.64660000e+04, -8.00280941e+01],
            [4.27571700e+06, -8.13530869e+01],
            ])),
    bnc.Benchmark(
        name="known nets",
        plot_front=False,
        marker="+",
        color="tab:purple",
        data=np.array([
            [11.69E6, -93.87],
            [25.56E6, -87.99],
            [44.55E6, -90.41],
            [61.10E6, -90.20],
        ]))
    ]

# Load dataset
here = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(here, "..", "datasets", "micro16flower"))
n_splits = 3
batch_size = 282
img_dim = 16  # Images will be resized to this value
img_shape = (3, img_dim, img_dim)
full_dataset = bnc.load_dataset(data_dir=data_dir, img_resize_dim=img_dim)
cross_valid = sklearn.model_selection.StratifiedKFold(
    n_splits=n_splits,
    random_state=42,
    shuffle=True,
    )

# Output directory. Change as desired.
output_dir = os.path.join(here, "..", "..", "bulkandcut_output", str(datetime.now()))

# Budget in seconds (also provided by the project)
budget = 86400.
budget_per_split = budget / n_splits

print("Initiating Evolution on device", bnc.device, "\n")
for s, (train_idx, valid_idx) in enumerate(cross_valid.split(full_dataset, full_dataset.targets)):
    print(f"Iniating training on split {s + 1} of {n_splits}")

    # Split dataset
    train_data = torch.utils.data.Subset(dataset=full_dataset, indices=train_idx)
    valid_data = torch.utils.data.Subset(dataset=full_dataset, indices=valid_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        )

    # Run a full optimization:
    work_dir = os.path.join(output_dir, f"split_{s+1}")
    evolution = bnc.Evolution(
        input_shape=img_shape,
        n_classes=len(full_dataset.classes),
        work_directory=work_dir,
        train_data_loader=train_loader,
        valid_data_loader=valid_loader,
        debugging=True,   # If True, the model will be validated after each epoch and learning
                          # curves will be plotted. If False, models are validaded just after
                          # been fully trainned.  #TODO: change to False
        )
    evolution.run(time_budget=budget_per_split)

    # Generate Pareto front plots and animation.
    # The time spent on plotting those images wont be deducted from the budget.
    # So, be free to comment this out if desired.
    bnc.generate_pareto_animation(
        working_dir=work_dir,
        ref_point=ref_point,
        benchmarks=benchmarks,
        )

    break  #TODO: remove
