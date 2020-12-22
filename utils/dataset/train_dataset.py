import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np


class TrainDataset(Dataset):
    """
    Dataset class enabling training model and handling all transforms.

    Example dataset structure:
    ├───data
    ├───train
    │   ├───inputs
    │   │   ├─── 0.png
    │   │   └─── 1.png
    │   └───masks
    │       ├───class1
    │       │   ├─── 0.png
    │       │   └─── 1.png
    │       └───class2
    │           ├─── 0.png
    │           └─── 1.png
    └───validate
        ├───inputs
        │   └─── test0.png
        └───masks
            ├───class1
            │   └─── test0.png
            └───class2
                └─── test0.png

    Inputs images are represented as tensors of size [number of channels, 256, 256].
    Masks are represented as tensors of size [number of classes, 256, 256].
    Values inside this tensors are transformed to floats from range [0; 1]. Because of that we can interpret masks
    as type I fuzzy sets.
    """

    def __init__(self, dir_path: str, as_gray: bool = False, size: int = 256):
        """
        Initialize dataset, setup inputs, classes and basic parameters
        :param dir_path: path to directory
        :param as_gray: whether images has to be converted to gray
        :param size: size of image (images will be resized to size x size)
        """
        self.__inputs = glob.glob(os.path.join(dir_path, 'inputs', '*'))
        self.__classes = glob.glob(os.path.join(dir_path, 'masks', '*'))
        self.__as_gray = as_gray
        self.__size = size

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        """
        Returns tuple of (input image, tensor of masks)
        :param item: index of item
        :return: tuple (input image, tensor of masks)
        """
        img_path = self.__inputs[item]
        masks_tensor = self.__get_masks(img_path)
        input_tensor = self.__get_input(img_path)
        return input_tensor, masks_tensor

    def __len__(self) -> int:
        """
        Returns length of inputs (how many inputs images in dataset)
        :return: length of self.__inputs
        """
        return len(self.__inputs)

    @property
    def classes(self) -> list:
        """
        Returns list of classes (name of directories inside "mask" dir)
        :return: list of classes
        """
        return [os.path.basename(class_dir) for class_dir in self.__classes]

    def __get_masks(self, img_path: str) -> torch.Tensor:
        """
        Gets all mask corresponding to certain image
        :param img_path: path to input image
        :return: tensor of masks
        """
        masks = []
        for mask_dir in self.__classes:
            mask_path = os.path.join(mask_dir, os.path.basename(img_path))
            img = imread(mask_path, as_gray=True)
            img_tensor = self.__mask_transform(img)
            masks.append(img_tensor)
        mask_tensor = torch.cat(masks, dim=1)
        return mask_tensor

    def __get_input(self, img_path: str) -> torch.Tensor:
        """
        Gets input image
        :param img_path: path to image
        :return: transformed image (as torch.Tensor)
        """
        img = imread(img_path, as_gray=self.__as_gray)
        return self.__input_transform(img)

    def __base_transform(self, x: np.ndarray) -> torch.Tensor:
        """
        Base transforms performed for both input images and masks:
        - resize (to default size 256x256)
        - represent images as matrix with values from range [0.; 1.] instead of [0; 255] uint
        - to float tensor
        :param x: image (as np.ndarray)
        :return: transformed image (as torch.Tensor)
        """
        x = resize(x, (self.__size, self.__size))
        x = torch.from_numpy(x).reshape((1, -1, self.__size, self.__size)).float()
        return x

    def __mask_transform(self, x: np.ndarray) -> torch.Tensor:
        """
        Perform transforms for mask
        :param x: input mask ( as np.ndarray)
        :return: transformed image (as torch.Tensor)
        """
        return self.__base_transform(x)

    def __input_transform(self, x: np.ndarray) -> torch.Tensor:
        """
        Perform transforms for input image
        :param x: input mask ( as np.ndarray)
        :return: transformed image (as torch.Tensor)
        """
        return self.__base_transform(x)


class TrainLoader(DataLoader):
    """
    Extends torch standard DataLoader, allowing batch size training and also holding
    information about number of classes and is empty.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.__empty = len(dataset) > 0
        self.__classes = dataset.classes

    @property
    def not_empty(self) -> bool:
        """
        Returns is dataset is empty (used especially to handle data loader)
        :return: whether dataset is empty
        """
        return self.__empty

    @property
    def classes(self) -> list:
        """
        Returns list of class names
        :return: list of classes (dir names in directory "masks")
        """
        return self.__classes

    @staticmethod
    def collate_data(batch) -> (torch.Tensor, torch.Tensor):
        """
        Collate dataset allowing to return tuple
        :param batch: given batch
        :return: tuple of tensors (images and masks)
        """
        images, masks = zip(*batch)
        return images, masks
