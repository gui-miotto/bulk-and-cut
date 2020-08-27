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


class BlindModel(torch.nn.Module):

    def __init__(self, n_classes:int):
        super(BlindModel, self).__init__()
        bias = torch.rand(n_classes) * 1E-6
        self.bias = torch.nn.Parameter(data=bias, requires_grad=True)
        self.loss_func_CE_hard = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=2.244958736283895e-05)
        self.n_classes = n_classes

    @property
    def n_parameters(self):
        return np.sum(par.numel() for par in self.parameters())

    @property
    def summary(self):
        model_summary = torchsummary.summary_string(
            model=self,
            input_size=(1,),
            device=device,
            )
        return model_summary[0]

    def save(self, file_path):
        torch.save(obj=self, f=file_path)


    def forward(self, x):
        batch_size = x.size(0)
        ones = torch.ones((batch_size, self.n_classes)).to(device)
        x = self.bias * ones # The blind model doesn't care about the input
        return x


    def start_training(
        self,
        train_data_loader: "torch.utils.data.DataLoader",
        valid_data_loader: "torch.utils.data.DataLoader",
        ):
        learning_curves = defaultdict(list)

        # Pre-training validation loss:
        print("Pre-training evaluation:")
        initial_loss, _ = self.evaluate(
            data_loader=valid_data_loader,
            split_name="validation",
            )
        learning_curves["validation_loss"].append(initial_loss)
        print("\n")

        train_batch_losses = self._train_one_epoch(train_data_loader=train_data_loader)
        learning_curves["train_loss"].append(train_batch_losses())

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

        status_str = f"training loss: {learning_curves['train_loss'][-1]:.3f}, "
        status_str += f"validation loss: {valid_loss:.3f}, "
        status_str += f"validation accuracy: {valid_accuracy:.3f}"
        print(status_str + "\n")

        return learning_curves

    def _train_one_epoch(self, train_data_loader):
        self.train()

        batch_losses = AverageMeter()
        tqdm_ = tqdm.tqdm(train_data_loader)
        for images, targets in tqdm_:
            batch_size =  images.size(0)
            images = images.to(device)
            targets = targets.to(device)

            # Forward- and backprop:
            self.optimizer.zero_grad()
            logits = self(images)
            loss = self.loss_func_CE_hard(input=logits, target=targets)
            loss.backward()
            self.optimizer.step()

            # Register training loss of the current batch:
            loss_value = loss.item()
            batch_losses.update(val=loss_value, n=batch_size)
            tqdm_.set_description(desc=f"Training loss: {loss_value:.3f}")

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
