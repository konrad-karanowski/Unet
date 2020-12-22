import os
import glob
import torch
from typing import List, NoReturn


from utils.model import setup_model
from utils.visualize import visualize_results


def get_files(input_path: str) -> List[str]:
    """
    Check if input path leads to single image or directory of images
    :param input_path: input path given by user
    :return: list of paths
    """
    if os.path.isdir(input_path):
        return glob.glob(os.path.join(input_path, '*'))
    else:
        return [input_path]


def predict_mask(as_gray: bool, num_classes: int, model_path: str,
                 input_path: str, threshold: float, output_path: str) -> NoReturn:
    """
    Predicts masks for given images
    :param as_gray: whether images has to be gray or coloured
    :param num_classes: number of classes we want to predict
    :param model_path: path to model weights
    :param input_path: path to input (single image or directory of images)
    :param threshold: threshold for which we want to classify pixel as belonging to the class
    :param output_path: path where we want save our predictions
    :return: NoReturn
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_model(1 if as_gray else 3, num_classes, model_path)
    model.to(device)
    images = get_files(input_path)
    visualize_results(model, device, as_gray, threshold, images, output_path)
