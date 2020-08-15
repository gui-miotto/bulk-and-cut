import os
import argparse
import logging
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold   # We use 3-fold stratified cross-validation

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel


def main(model_config,
         data_dir,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    #data_augmentations = [transforms.Resize([img_width, img_height]),
    #                      transforms.ToTensor()]
    if data_augmentations is None:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Load the dataset
    data = ImageFolder(data_dir, transform=data_augmentations)

    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []
    for train_idx, valid_idx in cv.split(data, data.targets):

        train_data = Subset(data, train_idx)
        test_dataset = Subset(data, valid_idx)

        #image size
        input_shape = (3, img_width, img_height)


        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        model = torchModel(model_config,
                           input_shape=input_shape,
                           num_classes=len(data.classes)).to(device)

        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        total_model_params = np.sum(p.numel() for p in model.parameters())
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS THE SECOND OBJECTIVE YOU HAVE TO OPTIMIZE
        # instantiate optimizer
        optimizer = model_optimizer(model.parameters(),
                                    lr=learning_rate)

        # Just some info for you to see the generated network.
        logging.info('Generated Network:')
        summary(model, input_shape,
                device='cuda' if torch.cuda.is_available() else 'cpu')

        # Train the model
        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            test_score = model.eval_fn(test_loader, device)

            logging.info('Split-Train accuracy %f', train_score)
            logging.info('Split-Test accuracy %f', test_score)
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
        score.append(test_score)
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE
        # THIS HERE IS PART OF THE FIRST OBJECTIVE YOU HAVE TO OPTIMIZE

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)

    # RESULTING SCORES FOR BOTH OBJECTIVES
    # RESULTING SCORES FOR BOTH OBJECTIVES
    # RESULTING SCORES FOR BOTH OBJECTIVES
    print('Resulting Model Score:')
    print('negative acc [%] | #num model parameters')
    print(1-np.mean(score), total_model_params)


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network.

    Also this contains the default configuration you should always capture with your
    configuraiton space!
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'mse': torch.nn.MSELoss}
    opti_dict = {'adam': torch.optim.Adam,
                 'adamw': torch.optim.AdamW,
                 'adad': torch.optim.Adadelta,
                 'sgd': torch.optim.SGD}

    cmdline_parser = argparse.ArgumentParser('AutoML SS20 final project')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=282,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'micro17flower'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=2.244958736283895e-05,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adamw',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # architecture parametrization
    architecture = {
        'n_conv_layers': 2,
        'n_channels_conv_0': 457,
        'n_channels_conv_1': 511,
        'n_channels_conv_2': 38,
        'kernel_size': 5,
        'global_avg_pooling': True,
        'use_BN': False,
        'n_fc_layers': 3,
        'n_channels_fc_0': 27,
        'n_channels_fc_1': 17,
        'n_channels_fc_2': 273}

    main(
        architecture,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        data_augmentations=None,  # Not set in this example
        save_model_str=args.model_path
    )
