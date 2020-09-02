import os
from datetime import datetime

import torch
import sklearn.model_selection

import bulkandcut as bnc


here = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(here, "..", "micro17flower"))
#work_dir = os.path.join("/tmp", "evolution_runs", str(datetime.now()))
work_dir = os.path.join(here, "..", "..", "evolution_runs", str(datetime.now()))


batch_size = 282
img_dim = 16  # Images will be resized to this value
img_shape = (3, img_dim, img_dim)

full_dataset = bnc.load_dataset(data_dir=data_dir, img_resize_dim=img_dim)





n_splits = 3

cross_valid = sklearn.model_selection.StratifiedKFold(
    n_splits=n_splits,
    random_state=42,
    shuffle=True,
    )

print("Initiating Evolution on device", bnc.device, "\n")

for train_idx, valid_idx in cross_valid.split(full_dataset, full_dataset.targets):

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

    evolution = bnc.Evolution(
        input_shape=img_shape,
        n_classes=len(full_dataset.classes),
        work_directory=work_dir,
        train_data_loader=train_loader,
        valid_data_loader=valid_loader,
        debugging=True,
        )


    start = datetime.now()
    #evolution.run(time_budget=5. * 60.)
    evolution.run(time_budget=8. * 60. * 60.)
    end = datetime.now()

    print((end - start).seconds / 60.)

    break
