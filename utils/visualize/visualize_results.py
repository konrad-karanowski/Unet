import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from typing import List, NoReturn


from utils.model import UNet


COLORS = [
    [235, 20, 60],
    [10, 20, 230],
    [0, 255, 80],
    [138, 43, 226],
    [255, 255, 0],
    [255, 105, 80]
]


def open_image(path: str, as_gray: bool) -> (np.ndarray, torch.Tensor):
    """
    Opens an image and do base preprocessing
    :param path: path to image
    :param as_gray: whether image has to be gray or coloured
    :return: image represented as np.ndarray and torch.Tensor
    """
    img = imread(path, as_gray=as_gray)
    img_resized = resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized).reshape(1, -1, 256, 256).float()
    return img_resized, img_tensor


def predict(model: UNet, device: torch.device, img: torch.Tensor) -> np.ndarray:
    """
    Predict output using UNet
    :param model: UNet model
    :param device: torch device
    :param img: input image
    :return: predictions as np.ndarray
    """
    img = img.to(device)
    output = model(img)
    return output.to(torch.device('cpu')).detach().numpy()


def cut_off_threshold(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Cut off
    :param img: predicted image
    :param threshold: threshold for which we want to classify pixel as belonging to the class
    :return: binary tensor of masks
    """
    return np.where(img >= threshold, 1., 0.)


def apply_mask_for_one(img: np.ndarray, mask: np.ndarray, color: List[int], alpha: float) -> np.ndarray:
    """
    Apply mask for one image
    :param img: image
    :param mask: binary mask
    :param color: color (in format [R, G, B])
    :param alpha: alpha parameter
    :return: image with applied one mask
    """
    for i, c in enumerate(color):
        img[:, :, i] = np.where(
            mask == 1,
            img[:, :, i] * (1 - alpha) + alpha * (c / 255),
            img[:, :, i]
        )
    return img


def apply_masks(img: np.ndarray, predictions: np.ndarray, colors: List[List[int]], alpha: float) -> np.ndarray:
    """
    Apply mask on image
    :param img: image
    :param predictions: tensor of masks
    :param colors: list of colors
    :param alpha: alpha parameter
    :return: image with applied masks
    """
    for i, mask in enumerate(predictions.reshape((-1, 256, 256))):
        img = apply_mask_for_one(img, mask, colors[i], alpha)
    return img


def plot_image(img: np.ndarray, predicted: np.ndarray, save_path: str or None) -> NoReturn:
    """
    Plots image and save if user want to
    :param img: base image
    :param predicted: predicted image
    :param save_path: path to save an image
    :return: NoReturn
    """
    img = apply_masks(img, predicted, COLORS, alpha=0.5)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_save_path(img_path: str, save_path) -> str:
    """
    Create path for saving image
    :param img_path: base image path
    :param save_path: path to save an image
    :return: new path to save an image
    """
    file = os.path.basename(img_path)
    output_name = f'{file.split(".")[0]}out.png'
    return os.path.join(save_path, output_name)


def visualize_results(model: UNet, device: torch.device, as_gray: bool,
                      threshold: float, images: List[str], save_path: str or None) -> NoReturn:
    """
    Predict images and visualize results
    :param model: UNet model
    :param device: torch device
    :param as_gray: whether images has to be gray or coloured
    :param threshold: threshold for which we want to classify pixel as belonging to the class
    :param images: list of paths to images
    :param save_path: path to save an image
    :return: NoReturn
    """
    for img_path in images:
        img, input_ = open_image(img_path, as_gray)
        prediction = predict(model, device, input_)
        result = cut_off_threshold(prediction, threshold)
        output_path = create_save_path(img_path, save_path) if save_path else None
        if as_gray:
            img = np.concatenate((img.reshape((256, 256, -1)),) * 3, axis=-1)
        else:
            img = img.reshape((256, 256, -1))
        plot_image(img, result, output_path)
