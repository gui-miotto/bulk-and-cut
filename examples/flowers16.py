import os
import argparse
from datetime import datetime

import torch
import torchvision
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

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Run bulkandcut on micro16flower")
    parser.add_argument(
        "out_path",
        metavar="output path",
        type=str,
        nargs="?",
        default=os.path.join(here, "..", "..", "bulkandcut_output", str(datetime.now())),
        help="Output directory for models, logs, plots, etc.")
    args = parser.parse_args()

    # Load dataset
    data_dir = os.path.abspath(os.path.join(here, "..", "datasets", "micro16flower"))
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=torchvision.transforms.ToTensor(),
        )
    n_splits = 3
    batch_size = 282
    img_dim = 16
    img_shape = (3, img_dim, img_dim)
    cross_valid = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        random_state=bnc.global_seed,
        shuffle=True,
        )

    # Budget in seconds (also provided by the project)
    budget = 86400.
    budget_per_split = budget / n_splits

    print("Initiating Evolution on device", bnc.device, "\n")
    for s, (tra_idx, val_idx) in enumerate(cross_valid.split(full_dataset, full_dataset.targets)):
        print(f"Iniating training on split {s + 1} of {n_splits}")

        # Split dataset
        train_data = torch.utils.data.Subset(dataset=full_dataset, indices=tra_idx)
        valid_data = torch.utils.data.Subset(dataset=full_dataset, indices=val_idx)
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
        work_dir = os.path.join(args.out_path, f"split_{s+1}")
        evolution = bnc.Evolution(
            input_shape=img_shape,
            n_classes=len(full_dataset.classes),
            work_directory=work_dir,
            train_data_loader=train_loader,
            valid_data_loader=valid_loader,
            debugging=False,
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
