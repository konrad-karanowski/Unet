import os
from typing import NoReturn, List
import torch
from torch import nn

from utils.model import setup_model
from utils.dataset import TrainDataset, TrainLoader


from utils.visualize import visualize_loss
from utils.summary_writer import SummaryWriter
from utils.metrics import MetricWriter


def setup_dataset(batch_size: int, as_gray: bool) -> (TrainLoader, TrainLoader, List[str]):
    """
    Prepares both training and validation datasets for training UNet
    :param batch_size: number of samples in batch
    :param as_gray: whether images has to be rescaled to gray
    :return: train loader, validate loader, classes
    """
    train_dataset = TrainDataset(os.path.join(os.curdir, 'data', 'train'), as_gray)
    validate_dataset = TrainDataset(os.path.join(os.curdir, 'data', 'validate'), as_gray)
    train_loader = TrainLoader(train_dataset, batch_size=batch_size)
    validate_loader = TrainLoader(validate_dataset, batch_size=batch_size)
    return train_loader, validate_loader, train_dataset.classes


def train_model(num_epochs: int, batch_size: int, lr: float, momentum: float,
                n_callbacks: int, model_path: str, as_gray: bool) -> NoReturn:
    """
    Train UNet model

    TODO: change criterion (?), add early stopping

    :param num_epochs: number of epochs
    :param batch_size: number of samples in batch
    :param lr: learning rate (eta)
    :param momentum: Nesterov momentum rate
    :param n_callbacks: how often user wants to save weights
    :param model_path: path to model weights if user wants to use pretrained one
    :param as_gray: whether images has to be rescaled to gray
    :return: NoReturn
    """
    train_loader, val_loader, classes = setup_dataset(batch_size, as_gray)
    num_classes = len(classes)
    model = setup_model(1 if as_gray else 3, num_classes, model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO Change criterion (?)
    criterion = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if num_classes > 1 else 'max', patience=2)
    model.to(device)

    SummaryWriter.write_model_summary(device, model)
    metric_writer = MetricWriter()
    # main training loop
    for i in range(1, num_epochs + 1):
        # training model
        for batch in train_loader:
            for input_, mask in zip(*batch):
                input_ = input_.to(device)
                mask = mask.to(device)
                output = model(input_)
                loss = criterion(output, mask)
                metric_writer.add_train_loss(loss.item())
                metric_writer.calculate_train_metrics(output, mask)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # validate model
        if val_loader.not_empty:
            for batch in val_loader:
                for input_, mask in zip(*batch):
                    with torch.no_grad():
                        input_ = input_.to(device)
                        mask = mask.to(device)
                        output = model(input_)
                        loss = criterion(output, mask)
                        metric_writer.add_val_loss(loss.item())
                        metric_writer.calculate_test_metrics(output, mask)
        else:
            metric_writer.add_val_loss(0)

        # save weights
        if i % n_callbacks == 0:
            torch.save(model.state_dict(), os.path.join('callbacks', f'Epoch{i}callback.pth'))
            pass
        # calculate metrics and write info
        mean_train_loss, mean_val_loss = metric_writer.close_epoch()
        SummaryWriter.write_epoch_info(i, num_epochs, mean_train_loss, mean_val_loss)
    torch.save(model.state_dict(), os.path.join('callbacks', 'model.pth'))
    visualize_loss(metric_writer.metrics)
