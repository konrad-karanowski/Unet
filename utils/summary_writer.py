import torch
from typing import Optional, NoReturn
from torchsummary import summary


class SummaryWriter:
    """
    Class handling all prints during the training process (they can be easily converted to logs)
    """

    @staticmethod
    def write_epoch_info(num_epoch: int, num_epochs: int, train_loss: list,
                         validate_loss: Optional[list] = None) -> NoReturn:
        """
        Writes info about number of epoch, train loss and validate loss
        :param num_epoch: number of epoch
        :param num_epochs: number of all epochs
        :param train_loss: mean train loss in epoch
        :param validate_loss: mean validate loss in epoch
        :return: NoReturn
        """
        msg = f'Epoch: {num_epoch}/{num_epochs}, training loss: {train_loss}'
        if validate_loss:
            msg = f'{msg} validate loss: {validate_loss}'
        print(msg)

    @staticmethod
    def write_model_summary(device: torch.device, model: torch.nn.Module) -> NoReturn:
        """
        Write summary of model using torchsummary library
        :param device: using device
        :param model: UNet model
        :return: NoReturn
        """
        print(f'Used device: {device}')
        print(f'Model summary: ')
        summary(model, model.input_size)

    @staticmethod
    def write_early_stopping(epochs: int) -> NoReturn:
        print(f'Early stopping after {epochs} epochs.')
