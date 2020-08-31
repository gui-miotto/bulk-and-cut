from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torchsummary
import tqdm
import matplotlib.pyplot as plt

from bulkandcut.model_section import ModelSection
from bulkandcut.model_head import ModelHead
from bulkandcut.dataset import mixup
from bulkandcut.average_meter import AverageMeter
from bulkandcut.cross_entropy_with_probs import CrossEntropyWithProbs
from bulkandcut import rng, device


class BNCmodel(torch.nn.Module):

    @classmethod
    def LOAD(cls, file_path:str) -> "BNCmodel":
        return torch.load(f=file_path).to(device)

    @classmethod
    def NEW(cls, input_shape, n_classes:int, optimizer_configuration:dict) -> "BNCmodel":
        # Sample
        n_conv_sections = rng.integers(low=1, high=4)

        # Convolutional layers
        conv_sections = torch.nn.ModuleList()
        in_elements = input_shape[0]
        for _ in range(n_conv_sections):
            conv_section = ModelSection.NEW(in_elements=in_elements, section_type="conv")
            in_elements = conv_section.out_elements
            conv_sections.append(conv_section)
        conv_sections[0].mark_as_first_section()

        # Fully connected (i.e. linear) layers
        linear_section = ModelSection.NEW(in_elements=in_elements, section_type="linear")
        head = ModelHead.NEW(
            in_elements=linear_section.out_elements,
            out_elements=n_classes,
        )

        return BNCmodel(
            conv_sections=conv_sections,
            linear_section=linear_section,
            head=head,
            input_shape=input_shape,
            optim_config=optimizer_configuration,
            ).to(device)


    def __init__(self, conv_sections, linear_section, head, input_shape, optim_config):
        super(BNCmodel, self).__init__()
        self.conv_sections = conv_sections
        self.glob_av_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_section = linear_section
        self.head = head
        self.input_shape = input_shape
        self.n_classes = head.out_elements

        self.loss_func_CE_soft = CrossEntropyWithProbs().to(device) #TODO: use the weights for unbalanced classes
        self.loss_func_CE_hard = torch.nn.CrossEntropyLoss().to(device)
        self.loss_func_MSE = torch.nn.MSELoss().to(device)
        self.optim_config = optim_config
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=10 ** optim_config["lr_exp"],
            weight_decay=10. ** optim_config["w_decay_exp"],
            )
        st_size = optim_config["lr_sched_step_size"] if "lr_sched_step_size" in optim_config else 1
        gamma = optim_config["lr_sched_gamma"] if "lr_sched_gamma" in optim_config else 1.
        self.LR_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=int(st_size),
            gamma=gamma,
            )


    @property
    def n_parameters(self):
        return np.sum(par.numel() for par in self.parameters())

    @property
    def summary(self):
        model_summary = torchsummary.summary_string(
            model=self,
            input_size=self.input_shape,
            device=device,
            )
        summary_str = model_summary[0] + "\n\n"
        summary_str += "Skip connections\n" + "-" * 30 + "\n"
        for cs, conv_sec in enumerate(self.conv_sections):
            summary_str += f"Convolutional section #{cs + 1}:\n"
            summary_str += conv_sec.skip_connections_summary
        summary_str += f"Linear section:\n"
        summary_str += self.linear_section.skip_connections_summary
        return summary_str

    def save(self, file_path):
        torch.save(obj=self, f=file_path)

    def forward(self, x):
        # convolutional cells
        for conv_sec in self.conv_sections:
            x = conv_sec(x)
        x = self.glob_av_pool(x)
        # flattening
        x = x.view(x.size(0), -1)
        # linear cells
        x = self.linear_section(x)
        x = self.head(x)
        return x

    def bulkup(self, optim_config) -> "BNCmodel":
        new_conv_sections = deepcopy(self.conv_sections)

        # There is a p chance of adding a convolutional cell
        if rng.uniform() < .7:
            sel_section = rng.integers(low=0, high=len(self.conv_sections))
            new_conv_sections[sel_section] = self.conv_sections[sel_section].bulkup()
            new_linear_section = deepcopy(self.linear_section)
        # And a (1-p) chance of adding a linear cell
        else:
            new_linear_section = self.linear_section.bulkup()

        new_head = self.head.bulkup()  # just returns a copy

        return BNCmodel(
            conv_sections=new_conv_sections,
            linear_section=new_linear_section,
            head=new_head,
            input_shape=self.input_shape,
            optim_config=optim_config,
            ).to(device)

    def slimdown(self, optim_config=None) -> "BNCmodel":
        # Prune head
        new_head, out_selected = self.head.slimdown(
            amount=rng.triangular(left=.04, right=.06, mode=.05),
            )
        # Prune linear section
        new_linear_section, out_selected = self.linear_section.slimdown(
            out_selected=out_selected,
            amount=rng.triangular(left=.065, right=.085, mode=.075),
            )
        # Prune convolutional sections
        new_conv_sections = torch.nn.ModuleList()
        for conv_sec in self.conv_sections[::-1]:
            new_section, out_selected = conv_sec.slimdown(
                out_selected=out_selected,
                amount=rng.triangular(left=.09, right=.11, mode=.10),
                )
            new_conv_sections.append(new_section)

        return BNCmodel(
            conv_sections=new_conv_sections[::-1],
            linear_section=new_linear_section,
            head=new_head,
            input_shape=self.input_shape,
            optim_config=optim_config,
            ).to(device)


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
            train_loss = self._train_one_epoch(
                train_data_loader=train_data_loader,
                teacher_model=teacher_model,
                loss_function=loss_func,
                )

            # Register perfomance of the current epoch:
            learning_curves["train_loss"].append(train_loss)
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
            images, labels = mixup(data=images, targets=labels, n_classes=self.n_classes)
            images = images.to(device)

            # If a teacher model was given, we use its predictions as targets,
            # otherwise we stick to the image labels.
            if teacher_model is not None:
                targets = teacher_model(images)
                targets = targets.to(device)
            else:
                targets = labels.to(device)

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
        return batch_losses()


    @torch.no_grad()
    def evaluate(self, data_loader, split_name):
        self.eval()

        average_loss = AverageMeter()
        average_accuracy = AverageMeter()
        tqdm_ = tqdm.tqdm(data_loader)
        for images, labels in tqdm_:
            batch_size =  images.size(0)

            # No mixup here!
            images = images.to(device)
            labels = labels.to(device)

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
