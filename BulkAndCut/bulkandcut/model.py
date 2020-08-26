from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torchsummary
import tqdm
import matplotlib.pyplot as plt

from bulkandcut.conv_cell import ConvCell
from bulkandcut.linear_cell import LinearCell
from bulkandcut.dataset import mixup
from bulkandcut.average_meter import AverageMeter
from bulkandcut.cross_entropy_with_probs import CrossEntropyWithProbs


class BNCmodel(torch.nn.Module):

    rng = np.random.default_rng(seed=1)  #TODO: this should come from above, so that we seed the whole thing (torch, numpy, cross-validation splits just at one place)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def LOAD(cls, file_path:str) -> "BNCmodel":  #TODO: this raising a lot of warnings. Why?
        return torch.load(f=file_path).to(BNCmodel.device)

    @classmethod
    def NEW(cls, input_shape, n_classes:int, optimizer_configuration:dict) -> "BNCmodel":
        # Sample
        n_conv_trains = cls.rng.integers(low=1, high=4)

        # Convolutional layers
        conv_trains = torch.nn.ModuleList()
        in_channels = input_shape[0]
        for _ in range(n_conv_trains):
            cc = ConvCell.NEW(in_channels=in_channels, rng=cls.rng)
            mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            conv_train = torch.nn.ModuleList([cc, mp])
            conv_trains.append(conv_train)
            in_channels = cc.out_channels

        # Fully connected (i.e. linear) layers
        linear_cell = LinearCell.NEW(in_features=in_channels, rng=cls.rng)
        head = torch.nn.Linear(
            in_features=linear_cell.out_features,
            out_features=n_classes,
        )
        linear_train = torch.nn.ModuleList([linear_cell, head])

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=input_shape,
            optim_config=optimizer_configuration,
            ).to(BNCmodel.device)


    def __init__(self, conv_trains, linear_train, input_shape, optim_config):
        super(BNCmodel, self).__init__()
        self.conv_trains = conv_trains
        self.glob_av_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_train = linear_train  #TODO: train is an overloaded term, change it to section
        self.input_shape = input_shape
        self.n_classes = self.linear_train[-1].out_features

        self.loss_func_CE_soft = CrossEntropyWithProbs().to(self.device) #TODO: use the weights for unbalanced classes
        self.loss_func_CE_hard = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_func_MSE = torch.nn.MSELoss().to(self.device)
        self.optim_config = optim_config
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=10 ** optim_config["lr_exp"],
            weight_decay=10. ** optim_config["w_decay_exp"],
            )
        self.LR_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=optim_config["lr_sched_step_size"],
            gamma=optim_config["lr_sched_gamma"],
            )


    @property
    def n_parameters(self):
        return np.sum(par.numel() for par in self.parameters())

    @property
    def summary(self):
        model_summary = torchsummary.summary_string(
            model=self,
            input_size=self.input_shape,
            device=BNCmodel.device,
            )
        return model_summary[0]

    def save(self, file_path):  #TODO: this raising a lot of warnings. Why?
        torch.save(obj=self, f=file_path)

    def forward(self, x):
        # convolutions and friends
        for ct in self.conv_trains:
            for module in ct:
                x = module(x)
        x = self.glob_av_pool(x)
        # flattening
        x = x.view(x.size(0), -1)
        # linear and friends
        for module in self.linear_train:
            x = module(x)
        return x

    def bulkup(self, optim_config=None) -> "BNCmodel":
        conv_trains = deepcopy(self.conv_trains)
        linear_train = deepcopy(self.linear_train)

        if BNCmodel.rng.uniform() < .7:  # There is a p chance of adding a convolutional cell
            sel_train = BNCmodel.rng.integers(
                low=0,
                high=len(conv_trains),
                )
            sel_cell = BNCmodel.rng.integers(
                low=0,
                high=len(conv_trains[sel_train]) - 1,  # Subtract 1 to exclude the maxpooling
                )
            identity_cell = conv_trains[sel_train][sel_cell].downstream_morphism()
            conv_trains[sel_train].insert(
                index=sel_cell + 1,
                module=identity_cell,
                )
        else:  # And a (1-p) chance of adding a linear cell
            sel_cell = BNCmodel.rng.integers(
                low=0,
                high=len(linear_train) - 1,  # Subtract 1 to exclude the head
                )
            identity_cell = linear_train[sel_cell].downstream_morphism()
            linear_train.insert(
                index=sel_cell + 1,
                module=identity_cell,
                )

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=self.input_shape,
            optim_config = self.optim_config if optim_config is None else optim_config,
            ).to(BNCmodel.device)

    def slimdown(self, optim_config=None) -> "BNCmodel":
        # Prune head
        linear_train = torch.nn.ModuleList()
        head, out_selected = self._prune_head()
        linear_train.append(head)

        # Prune linear cells
        for linear_cell in self.linear_train[-2::-1]:  # Reverse it and skip the first (net head)
            pruned_linear_cell, out_selected = linear_cell.prune(out_selected=out_selected)
            linear_train.insert(index=0, module=pruned_linear_cell)

        # Prune convolutional cells
        conv_trains = torch.nn.ModuleList()
        for ct in range(len(self.conv_trains))[::-1]:
            conv_train = self.conv_trains[ct]
            slimmer_conv_train = torch.nn.ModuleList()
            for cc in range(len(conv_train) - 1)[::-1]:  # Reverse it and skip the first (max pool)
                conv_cell = conv_train[cc]
                pruned_conv_cell, out_selected = conv_cell.prune(
                    out_selected=out_selected,
                    is_input_layer=ct==0 and cc==0,
                    )
                slimmer_conv_train.insert(index=0, module=pruned_conv_cell)
            maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            slimmer_conv_train.append(maxpool)
            conv_trains.insert(index=0, module=slimmer_conv_train)

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=self.input_shape,
            optim_config= self.optim_config if optim_config is None else optim_config,
            ).to(BNCmodel.device)


    def _prune_head(self):
        amount = .05
        parent_head = self.linear_train[-1]

        num_in_features = int((1. - amount) * parent_head.in_features)
        head = torch.nn.Linear(
            in_features=num_in_features,
            out_features=self.n_classes,
        )

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(parent_head.weight),
            dim=0,
        )
        in_selected = torch.argsort(w_l1norm)[-num_in_features:]
        in_selected = torch.sort(in_selected).values  # this is actually not not necessary

        weight = deepcopy(parent_head.weight.data[:,in_selected])
        bias = deepcopy(parent_head.bias)
        head.weight = torch.nn.Parameter(weight)
        head.bias = torch.nn.Parameter(bias)
        return head, in_selected


    def start_training(
        self,
        n_epochs:int,
        train_data_loader: "torch.utils.data.DataLoader",
        valid_data_loader: "torch.utils.data.DataLoader",
        teacher_model: "BNCmodel" = None,
        return_all_learning_curvers: bool = False,
        ):
        learning_curves = defaultdict(list)

        # If a parent model was provided, its logits will be used as targets (knowledge
        # distilation). In this case we are going to use a simple MSE as loss function.
        loss_func = self.loss_func_CE_soft if teacher_model is None else self.loss_func_MSE

        # Pre-training validation loss:
        print("Pre-training evaluation:")
        initial_loss, _ = self.evaluate(
            data_loader=valid_data_loader,
            split_name="validation",
            )
        learning_curves["validation_loss"].append(initial_loss)
        print("\n")

        for epoch in range(1, n_epochs + 1):
            train_batch_losses = self._train_one_epoch(
                train_data_loader=train_data_loader,
                teacher_model=teacher_model,
                loss_function=loss_func,
                )

            # Register perfomance of the current epoch:
            learning_curves["train_loss"].append(train_batch_losses())
            status_str = f"Epoch {epoch} results -- "
            status_str += f"learning rate: {self.LR_schedule.get_last_lr()[0]:.3e}, "
            status_str += f"training loss: {learning_curves['train_loss'][-1]:.3f}, "
            if return_all_learning_curvers or epoch == n_epochs:
                # If required, I'm going to monitor all sorts of learning curves,
                # otherwise I'll measure performance just once after the last epoch.
                train_loss_at_eval, train_accuracy = self.evaluate(
                    data_loader=train_data_loader,
                    split_name="training",
                    )
                valid_loss, valid_accuracy = self.evaluate(
                    data_loader=valid_data_loader,
                    split_name="validation",
                    )
                learning_curves["train_loss_at_eval"].append(train_loss_at_eval)
                learning_curves["train_accuracy"].append(train_accuracy)
                learning_curves["validation_loss"].append(valid_loss)
                learning_curves["validation_accuracy"].append(valid_accuracy)

                status_str += f"validation loss: {valid_loss:.3f}, "
                status_str += f"validation accuracy: {valid_accuracy:.3f}"
            print(status_str + "\n")

        return learning_curves

    def _train_one_epoch(self, train_data_loader, teacher_model, loss_function):
        self.train()
        if teacher_model is not None:
            teacher_model.eval()

        batch_losses = AverageMeter()
        tqdm_ = tqdm.tqdm(train_data_loader)
        for images, labels in tqdm_:
            batch_size =  images.size(0)

            # Apply mixup
            images, labels = mixup(
                data=images,
                targets=labels,
                n_classes=self.n_classes,
                rng=BNCmodel.rng,
            )
            images = images.to(BNCmodel.device)

            # If a teacher model was given, we use its predictions as targets,
            # otherwise we stick to the image labels.
            if teacher_model is not None:
                targets = teacher_model(images)
                targets = targets.to(BNCmodel.device)
            else:
                targets = labels.to(BNCmodel.device)

            # Forward- and backprop:
            self.optimizer.zero_grad()
            logits = self(images)
            loss = loss_function(input=logits, target=targets)
            loss.backward()
            self.optimizer.step()

            # Register training loss of the current batch:
            loss_value = loss.item()
            batch_losses.update(val=loss_value, n=batch_size)
            tqdm_.set_description(desc=f"Training loss: {loss_value:.3f}")

        self.LR_schedule.step()
        return batch_losses


    @torch.no_grad()
    def evaluate(self, data_loader, split_name):
        self.eval()

        average_loss = AverageMeter()
        average_accuracy = AverageMeter()
        tqdm_ = tqdm.tqdm(data_loader)
        for images, labels in tqdm_:
            batch_size =  images.size(0)

            # No mixup here!
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Loss:
            logits = self(images)
            loss_value = self.loss_func_CE_hard(input=logits, target=labels)
            average_loss.update(val=loss_value.item(), n=batch_size)

            # Top-3 accuracy:
            top3_accuracy = self._accuracy(outputs=logits, targets=labels, topk=(3,))
            average_accuracy.update(val=top3_accuracy[0], n=batch_size)

            tqdm_.set_description(f"Evaluating on the {split_name} split:")

        return average_loss(), average_accuracy()


    @torch.no_grad()
    def _accuracy(self, outputs, targets, topk=(1,)):
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.T
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            accuracies.append(correct_k.mul_(100.0/batch_size).item())
        return accuracies
