import torch
from typing import NoReturn

from utils.metrics.metrics import *


class MetricWriter:
    """
    Practical structure to hold all info about metrics

    Params:
    __metrics: metrics values in successive epochs
    __epoch_metrics: metrics during the epoch
    """
    def __init__(self):
        """
        Initialize two dictionaries, __metrics and __epoch_metrics
        """
        self.__metrics = {
            'train_loss': [],
            'validate_loss': [],
            'train_iou': [],
            'validate_iou': [],
            'train_dice': [],
            'validate_dice': [],
            'train_alternative_loss': [],
            'validate_alternative_loss': []
        }

        self.__epoch_metrics = {
            'train_loss': [],
            'validate_loss': [],
            'train_iou': [],
            'validate_iou': [],
            'train_dice': [],
            'validate_dice': [],
            'train_alternative_loss': [],
            'validate_alternative_loss': []
        }

    def add_train_loss(self, loss: torch.Tensor) -> NoReturn:
        """
        Add loss value to train loss in epoch
        :param loss: value of loss in batch
        :return: NoReturn
        """
        self.__epoch_metrics['train_loss'].append(loss)

    def add_val_loss(self, loss: torch.Tensor or int) -> NoReturn:
        """
        Add loss value to validation loss in epoch
        :param loss: value of loss in batch
        :return: NoReturn
        """
        self.__epoch_metrics['validate_loss'].append(loss)

    def calculate_train_metrics(self, output: torch.Tensor, target: torch.Tensor) -> NoReturn:
        iou, dice, alt_loss = self.__calculate_metrics(output, target)
        self.__epoch_metrics['train_iou'].append(iou)
        self.__epoch_metrics['train_dice'].append(dice)
        self.__epoch_metrics['train_alternative_loss'].append(alt_loss)

    def calculate_test_metrics(self, output: torch.Tensor, target: torch.Tensor):
        iou, dice, alt_loss = self.__calculate_metrics(output, target)
        self.__epoch_metrics['validate_iou'].append(iou)
        self.__epoch_metrics['validate_dice'].append(dice)
        self.__epoch_metrics['validate_alternative_loss'].append(alt_loss)

    def close_epoch(self) -> (float, float):
        """
        Calculates mean values of metrics and returns mean train loss and mean val loss this epoch
        :return: mean train loss this epoch, mean val loss this epoch
        """
        for item, result in self.__epoch_metrics.items():
            mean_result = sum(result) / (len(result) + 1e-8)
            self.__metrics[item].append(mean_result)
            result.clear()
        return self.train_loss[-1], self.validate_loss[-1]

    def __calculate_metrics(self, output: torch.Tensor,
                            target: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate metrics and returns it
        :param output: output of the model
        :param target: true masks
        :return: iou, dice coefficient and alternative loss
        """
        iou, intersection, union = fuzzy_iou(output, target)
        dice = dice_metric(intersection, union)
        alt_loss = alternative_loss(dice)
        return iou, dice, alt_loss

    @property
    def metrics(self) -> dict:
        """
        Returns all metrics in successive epochs
        :return: dictionary of metrics
        """
        return self.__metrics

    @property
    def train_loss(self) -> list:
        """
        Returns list with train loss in successive epochs
        :return: train loss
        """
        return self.__metrics['train_loss']

    @property
    def validate_loss(self) -> list:
        """
        Returns list with val loss in successive epochs
        :return: val loss
        """
        return self.__metrics['validate_loss']
