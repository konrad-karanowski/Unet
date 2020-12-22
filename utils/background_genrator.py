import os
import glob
import numpy as np
from skimage.io import imread, imsave
from typing import List, NoReturn


def get_images(base_dir: str) -> List[str]:
    """
    Gets images from certain dataset
    :param base_dir: name of general dataset dir 'train' or 'validate'
    :return: list of paths to images
    """
    return glob.glob(os.path.join('data', base_dir, 'inputs', '*'))


def get_classes(base_dir: str) -> List[str]:
    """
    Gets classes from certain dataset
    :param base_dir: name of general dataset dir 'train' or 'validate'
    :return: list of classes
    """
    return glob.glob(os.path.join('data', base_dir, 'masks', '*'))


def generate_background(img_path: str, classes: str, background_path: str) -> NoReturn:
    """
    Generate background and save it
    :param img_path: path to certain images
    :param classes: list of class names
    :param background_path: background to save image
    :return: NoReturn
    """
    img_shape = imread(img_path).shape[:2]
    background = np.ones(img_shape)
    img_basename = os.path.basename(img_path)
    for mask_path in classes:
        img = os.path.join(mask_path, img_basename)
        img = imread(img, as_gray=True)
        background = np.where(img >= 0.9, 0, background)
    imsave(os.path.join(background_path, img_basename), background)


def generate_for(name_dir: str) -> NoReturn:
    """
    Generate backgrounds masks for certain dataset
    :param name_dir: name of general dataset dir 'train' or 'validate'
    :return: NoReturn
    """
    images = get_images(name_dir)
    classes = get_classes(name_dir)
    background_path = os.path.join('data', name_dir, 'masks', 'background')
    if len(images) > 0:
        try:
            os.mkdir(background_path)
        except FileExistsError:
            print('Background already exists. Create only files inside it.')
        for i, img_path in enumerate(images):
            print(f'Created background: {i + 1} / {len(images)} in {name_dir} dataset.')
            generate_background(img_path, classes, background_path)
    else:
        print(f'{name_dir.capitalize()} dataset is empty.')


def generate_background_class() -> NoReturn:
    """
    Generate background masks for train and validate datasets
    :return: NoReturn
    """
    generate_for('train')
    generate_for('validate')
