import logging
from copy import deepcopy

import numpy as np
import torch
import torchsummary
import tqdm


from bulkandcut.conv_cell import ConvCell
from bulkandcut.linear_cell import LinearCell
from bulkandcut.dataset import mixup
from bulkandcut.average_meter import AverageMeter
from bulkandcut.cross_entropy_with_probs import CrossEntropyWithProbs


class BNCmodel(torch.nn.Module):

    rng = np.random.default_rng(seed=1)  #TODO: this should come from above, so that we seed the whole thing (torch, numpy, cross-validation splits just at one place)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def LOAD(cls, file_path):  #TODO: this raising a lot of warnings. Why?
        return torch.load(f=file_path)

    @classmethod
    def NEW(cls, input_shape, n_classes):
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
        conv_outputs = cls._get_conv_output(shape=input_shape, conv_trains=conv_trains)
        linear_cell = LinearCell.NEW(in_features=conv_outputs, rng=cls.rng)
        head = torch.nn.Linear(
            in_features=linear_cell.out_features,
            out_features=n_classes,
        )
        linear_train = torch.nn.ModuleList([linear_cell, head])

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=input_shape,
            conv_outputs=conv_outputs,
            )


    @classmethod  #TODO: move? maybe together with the accuracy
    @torch.no_grad()
    def _get_conv_output(cls, shape, conv_trains):
        bs = 1
        x = torch.rand(bs, *shape)
        for ct in conv_trains:
            for module in ct:
                x = module(x)
        n_size = x.data.view(bs, -1).size(1)
        return n_size


    def __init__(self, conv_trains, linear_train, input_shape, conv_outputs):
        super(BNCmodel, self).__init__()
        self.conv_trains = conv_trains
        self.linear_train = linear_train
        self.input_shape = input_shape
        self.conv_outputs = conv_outputs

        self.loss_func_CE_softlabels = CrossEntropyWithProbs().to(self.device) #TODO: use the weights for unbalanced classes
        self.loss_func_CE_hardlabels = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_func_MSE = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)  #TODO: dehardcode

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

    def save(self, file_path):
        torch.save(obj=self, f=file_path)

    def forward(self, x):
        # convolutions and friends
        for ct in self.conv_trains:
            for module in ct:
                x = module(x)
        # flattening
        x = x.view(x.size(0), -1)
        # linear and friends
        for module in self.linear_train:
            x = module(x)
        return x

    def bulkup(self):
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
            conv_outputs=self.conv_outputs,
            )

    #TODO: this is a backward prune, which doesn't make much sense. Implement the forward prune
    def slimdown(self):
        # self.input_shape[0] is the number of channels in the images
        in_select = list(range(self.input_shape[0]))

        # Prune convolutional cells
        conv_trains = torch.nn.ModuleList()
        for conv_train in self.conv_trains:
            slimmer_conv_train = torch.nn.ModuleList()
            for conv_cell in conv_train[:-1]:  # Subtract 1 to exclude the maxpool
                pruned_conv_cell, in_select = conv_cell.prune(in_select=in_select)
                slimmer_conv_train.append(pruned_conv_cell)
            maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            slimmer_conv_train.append(maxpool)
            conv_trains.append(slimmer_conv_train)

        # Convert filter index to unit index
        in_select = self._flatten_filter_index(in_select)

        # Prune linear cells
        linear_train = torch.nn.ModuleList()
        for linear_cell in self.linear_train[:-1]:  # Subtract 1 to exclude the head
            pruned_linear_cell, in_select = linear_cell.prune(in_select=in_select)
            linear_train.append(pruned_linear_cell)

        # Prune head (just incomming weights, not units, of course)
        head = self._prune_head(in_select=in_select)
        linear_train.append(head)

        return BNCmodel(
            conv_trains=conv_trains,
            linear_train=linear_train,
            input_shape=self.input_shape,
            conv_outputs=BNCmodel._get_conv_output(self.input_shape, conv_trains)
            )

    def _prune_head(self, in_select):
        parent_head = self.linear_train[-1]
        n_classes = parent_head.out_features

        head = torch.nn.Linear(
            in_features=len(in_select),
            out_features=n_classes,
        )
        weight = deepcopy(parent_head.weight.data[:,in_select]) # TODO: do I need this deep copy here?
        bias = deepcopy(parent_head.bias) # TODO: do I need this deep copy here?
        head.weight = torch.nn.Parameter(weight)
        head.bias = torch.nn.Parameter(bias)

        return head


    def _flatten_filter_index(self, selection):
        n_filters = self.conv_trains[-1][-2].out_channels
        # sanity check
        if self.conv_outputs % n_filters != 0.0:
            raise Exception("Wrong number of filters")
        upf = self.conv_outputs // n_filters  # units per filter

        # TODO: I'm not quite sure that this corresponds to the way the convolution output is flattten.
        # In other words, am I really selecting the weights of the units I want to keep???
        linear_selection = []
        for s in selection.numpy():
            linear_selection += list(range(s * upf, (s+1) * upf))
        return linear_selection

    def train_heavylift(self, n_epochs, train_data_loader, valid_data_loader):
        returnables = {}
        train_epoch_losses = []
        valid_epoch_losses = []
        valid_epoch_accuracies = []

        # Pre-training validation loss:
        returnables["pre_training_loss"], _ = self.evaluate(valid_data_loader)

        for epoch in range(n_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
            train_batch_losses = AverageMeter()

            tqdm_ = tqdm.tqdm(train_data_loader)
            for images, labels in tqdm_:
                batch_size =  images.size(0)

                # Apply mixup
                images, labels = mixup(
                    data=images,
                    targets=labels,
                    n_classes=17,  #TODO: de-hardcode it
                    rng=BNCmodel.rng,
                )
                images = images.to(BNCmodel.device)
                labels = labels.to(BNCmodel.device)

                # Forward- and backprop:
                self.train()
                self.optimizer.zero_grad()
                logits = self(images)
                loss = self.loss_func_CE_softlabels(input=logits, target=labels)
                loss.backward()
                self.optimizer.step()

                # Register training loss of the current batch:
                loss_value = loss.item()
                train_batch_losses.update(val=loss_value, n=batch_size)
                tqdm_.set_description(desc=f"(=> Training) Loss: {loss_value:.4f}")

            # Register perfomance of the current epoch:
            train_epoch_losses.append(train_batch_losses())
            valid_loss, valid_accuracy = self.evaluate(valid_data_loader)  # TODO: remove this when running for real. Evaluate just once at the end
            valid_epoch_losses.append(valid_loss)
            valid_epoch_accuracies.append(valid_accuracy)
            print(f"Epoch {epoch + 1}: training loss: {train_epoch_losses[-1]:.4f}, validation loss: {valid_loss:.4f}, validation accuracy: {valid_accuracy:.4f}")

        returnables["post_training_loss"] = valid_epoch_losses[-1]
        returnables["post_training_accuracy"] = valid_epoch_accuracies[-1]

        #TODO: delete or move
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(train_epoch_losses, label="train", color=color)
        ax1.plot(valid_epoch_losses, label="valid", color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(valid_epoch_accuracies, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.savefig("bulkingup.png")
        #plt.show()

        return returnables

    def train_cardio(self, n_epochs, parent_model, train_data_loader, valid_data_loader):
        returnables = {}
        train_epoch_losses = []
        valid_epoch_losses = []
        valid_epoch_accuracies = []

        # Pre-training validation loss:
        returnables["pre_training_loss"], _ = self.evaluate(valid_data_loader)

        for epoch in range(n_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
            train_batch_losses = AverageMeter()

            tqdm_ = tqdm.tqdm(train_data_loader)
            for images, labels in tqdm_:
                batch_size =  images.size(0)

                # Apply mixup.
                # Notice that we don't care about the real labels here
                images, _ = mixup(
                    data=images,
                    targets=labels,
                    n_classes=17,  #TODO: de-hardcode it
                    rng=BNCmodel.rng,
                )
                images = images.to(BNCmodel.device)

                # Get targets from the parent model
                parent_model.eval()
                targets = parent_model(images)

                # Foward and backprop:
                self.train()
                self.optimizer.zero_grad()
                logits = self(images)
                loss = self.loss_func_MSE(input=logits, target=targets)
                loss.backward()
                self.optimizer.step()

                # Register training loss of the current batch:
                loss_value = loss.item()
                train_batch_losses.update(val=loss_value, n=batch_size)
                tqdm_.set_description(desc=f"(=> Training) Loss: {loss_value:.4f}")

            # Register perfomance of the current epoch:
            train_epoch_losses.append(train_batch_losses())
            valid_loss, valid_accuracy = self.evaluate(valid_data_loader)  # TODO: remove this when running for real. Evaluate just once at the end
            valid_epoch_losses.append(valid_loss)
            valid_epoch_accuracies.append(valid_accuracy)
            print(f"Epoch {epoch + 1}: training loss: {train_epoch_losses[-1]:.4f}, validation loss: {valid_loss:.4f}, validation accuracy: {valid_accuracy:.4f}")

        returnables["post_training_loss"] = valid_epoch_losses[-1]
        returnables["post_training_accuracy"] = valid_epoch_accuracies[-1]

        #TODO: delete or move
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(train_epoch_losses, label="train", color=color)
        ax1.plot(valid_epoch_losses, label="valid", color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(valid_epoch_accuracies, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.savefig("slimmingdown.png")
        #plt.show()

        return returnables



    @torch.no_grad()
    def evaluate(self, valid_data_loader):
        self.eval()

        average_loss = AverageMeter()
        average_accuracy = AverageMeter()
        tqdm_ = tqdm.tqdm(valid_data_loader)
        for images, labels in tqdm_:
            batch_size =  images.size(0)

            # No mixup here!
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Loss:
            logits = self(images)
            loss_value = self.loss_func_CE_hardlabels(input=logits, target=labels)
            average_loss.update(val=loss_value.item(), n=batch_size)

            # Top-3 accuracy:
            top3_accuracy = accuracy(outputs=logits, targets=labels, topk=(3,))
            average_accuracy.update(val=top3_accuracy[0], n=batch_size)

            tqdm_.set_description("Evaluating on the validation split:")

        return average_loss(), average_accuracy()


#TODO: move this somewhere else?
def accuracy(outputs, targets, topk=(1,)):
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
