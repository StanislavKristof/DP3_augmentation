"""This file is responsible for loading of the data.

Taken from:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""


import warnings
from image_manipulation import increase_size
import pandas as pd
from hydra.utils import instantiate
import numpy as np
import torch
from typing import Optional, List
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
from icecream import ic
from pathlib import Path
import cv2 as cv
with warnings.catch_warnings(action="ignore"):
    from datasets import load_dataset
from utils import get_data_file_path
import albumentations as A
import matplotlib.pyplot as plt


def convert_image_tensor_to_float(image: np.ndarray) -> torch.Tensor:
    """Convert image tensor to float64.

    Parameters:
    image (np.ndarray):
        image represented as np.ndarray

    Returns:
    image converted to torch tensor with dtype float
    """
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)  # convert to C, H, W
    return image


# source:
# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CroppedSVHNDataset(Dataset):
    """Represents the cropped version of the SVHN Dataset.

    The Street View House Numbers (SVHN) Dataset is a real-world image dataset
    for classification with 10 classes (one for each digit). The cropped
    version of the dataset is used in this work, since we only have to deal
    with classification tasks.  All images have the resolution of 32 by 32. The
    sides of the images can contain distracting digits.

    In total, the dataset contains 630420 images, split like this:
        train = 73257
        test = 26032
        extra = 531131

    Source: http://ufldl.stanford.edu/housenumbers/

    Downloaded at:
        train_32x32.mat = 2025/02/02/12/05
        test_32x32.mat = 2025/02/02/12/06
        extra_32x32.mat = 2025/02/02/12/07
    """

    def __init__(
            self,
            dataset_path: str,
            ) -> None:
        """Construct a CropppedSVHNDataset object.

        Constructs a CroppedSVHNDataset object using the specified dataset path.

        Parameters:
            dataset_path (str): path to the .mat file
        """
        self.data = loadmat(dataset_path)
        # self.length = 10000
        self.length = self.data["X"].shape[-1]

    def __len__(self) -> int:
        """Obtain length of the data.

        Returns:
            int: quantity of the data
        """
        return self.length

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, int]:
        """Get item of the dataset.

        Parameters:
            idx (int): index of the datapoint in the dataset

        Returns:
            a tuple containing an image and label
        """
        image = self.data["X"][:, :, :, idx] / 255
        label = self.data["y"][idx][0] # only contains one item
        label = label - 1 # so that all numbers are in between 0 and 9

        # plt.imshow(image)
        # plt.show()
        # input()
        image = np.transpose(image, (2, 0, 1))

        return image, label, idx


class CIFAR10Dataset(Dataset):
    """Represents the CIFAR-10 Dataset.

    The CIFAR-10 Dataset consists of 60000 images in 10 classes, with 6000
    images per class. There are 50000 training images and 10000 test images.
    All images have the resolution of 32 by 32 and 3 channels (RGB).

    The dataset is split like this:
        data_batch_1 ... data_batch_5 = serves as train set
        test_batch = serves as test set
        batches.meta = meta data (mappings between class number and class name)

    Each of the files contains 10000 images.

    Source: https://www.cs.toronto.edu/~kriz/cifar.html

    Downloaded at:
        cifar-10-python.tar.gz = 2025/02/02/11/44
    """

    def __init__(
            self,
            dataset_path: str,
            dataset_files: List[str]
            ) -> None:
        """Construct a CIFAR10Dataset object.

        Constructs a CIFAR10Dataset object using the specified dataset path.

        Parameters:
            dataset_path (str): path to the datafile
        """
        dataset_path = Path(dataset_path)

        self.data = [
            unpickle(dataset_path / dataset_file)[b'data']
            for dataset_file in dataset_files
        ]

        self.labels = [
            unpickle(dataset_path / dataset_file)[b'labels']
            for dataset_file in dataset_files
        ]

        self.data = np.concatenate(self.data) / 255

        self.labels = np.concatenate(self.labels)

        self.length = self.labels.shape[0]

    def __len__(self) -> int:
        """Obtain length of the data.

        Returns:
            int: quantity of the data
        """
        return self.length

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, int]:
        """Get item of the dataset.

        Parameters:
            idx (int): index of the datapoint in the dataset

        Returns:
            a tuple containing an image and label
        """
        image = self.data[idx]
        image = image.reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)
        # plt.imshow(image)
        # plt.show()
        image = image.transpose(2, 0, 1)

        label = self.labels[idx]
        return image, label, idx


class CIFAR100Dataset(Dataset):
    """Represents the CIFAR-100 Dataset.

    The CIFAR-100 Dataset consists of 60000 images in 100 classes, with 600
    images per class. There are 50000 training images and 10000 test images. All
    images have the resolution of 32 by 32 and 3 channels (RGB). Additionally,
    each image has a superclass (20 of them), each superclass is divided into 5
    classes.

    The dataset is split like this:
        train = 50000 images
        test = 10000 images 
        meta = meta data (mappings between class number and class name)

    Each of the files contains 10000 images.

    Source: https://www.cs.toronto.edu/~kriz/cifar.html

    Downloaded at:
        cifar-100-python.tar.gz = 2025/02/10/13/19
    """

    def __init__(
            self,
            dataset_path: str,
            ) -> None:
        """Construct a CIFAR10Dataset object.

        Constructs a CIFAR10Dataset object using the specified dataset path.

        Parameters:
            dataset_path (str): path to the datafile
        """
        dataset_path = Path(dataset_path)

        self.data = unpickle(dataset_path)[b'data']

        self.fine_labels = unpickle(dataset_path)[b'fine_labels']
        self.coarse_labels = unpickle(dataset_path)[b'coarse_labels']

        self.data = self.data.astype("float32") / 255

        self.length = len(self.fine_labels)

        split = dataset_path.name
        if split == "train":
            self.transform = A.Compose([
                A.Sharpen(p=.2),
                #A.ColorJitter(p=.1, hue=(-.03, .03)), # hue=0
                #A.Emboss(p=.1),
                A.HueSaturationValue(p=.1, hue_shift_limit=(-10, 10)),
                A.ISONoise(p=.1),
                A.ImageCompression(p=.1, quality_range=(90, 95)),
                A.InvertImg(p=.05),
                #A.RGBShift(p=.15),
                A.OneOf([
                    A.Solarize(p=.5, threshold_range=(0.8, 0.8)),
                    A.Solarize(p=.5, threshold_range=(0.9, 0.9)),
                ], p=.1),

                #A.Superpixels(p=.1, max_size=12, n_segments=(2, 8)),

                A.OneOf([
                    A.ToGray(p=.6),
                    A.ToSepia(p=.4),
                ], p=.1),

                A.CoarseDropout(p=.1),
                A.HorizontalFlip(p=.1),
                A.Morphological(p=.15, operation="erosion", scale=3),
                A.Morphological(p=.15, operation="dilation", scale=3),
                A.PixelDropout(p=.1, per_channel=True),
                A.RandomRotate90(p=.1),
                A.RandomSizedCrop(p=.1, min_max_height=(28, 28), size=(32, 32)),
                A.Rotate(p=.1),
                A.Transpose(p=.1),
            ])
        elif split == "test":
            self.transform = A.Compose(A.NoOp(p=1))

    def __len__(self) -> int:
        """Obtain length of the data.

        Returns:
            int: quantity of the data
        """
        return self.length

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, int]:
        """Get item of the dataset.

        Parameters:
            idx (int): index of the datapoint in the dataset

        Returns:
            a tuple containing an image and label
        """
        image = self.data[idx]
        image = image.reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)
        # plt.imshow(image)
        # plt.show()

        fine_label = self.fine_labels[idx]

        # image = self.transform(image=image)["image"]
        image = image.transpose(2, 0, 1)
        # plt.imshow(image)
        # plt.show()

        return image, fine_label, idx


class StanfordCarsDataset(Dataset):
    """Represents the Stanford Cars Dataset.

    The Stanford Cars dataset contains 16185 images of 196 classes of cars. The
    class label usually consists of the make, model and year. Images have
    various sizes, some are RGB, some are greyscale.

    The dataset is split like this:
        train = 8144 images 
        test = 8041 images

    Source: https://huggingface.co/datasets/tanganke/stanford_cars

    Downloaded at: 2025/02/16/13/24
    """

    def __init__(
            self,
            dataset_path: str,
            ) -> None:
        """Construct a StanfordCarsDataset object.

        Constructs a StanfordCarsDataset object using the specified dataset
        path.

        Parameters:
            dataset_path (str): path to the datafile
        """
        self.dataset_path = Path(dataset_path)
        split = self.dataset_path.name
        dataset = load_dataset("tanganke/stanford_cars")
        self.dataset = dataset[split]
        self.length = len(self.dataset)
        if split == "train":
            self.transform = A.Compose([
                A.Sharpen(p=.2),
                #A.ColorJitter(p=.1, hue=(-.03, .03)), # hue=0
                #A.Emboss(p=.1),
                A.HueSaturationValue(p=.1, hue_shift_limit=(-10, 10)),
                A.ISONoise(p=.1),
                A.ImageCompression(p=.1, quality_range=(90, 95)),
                # A.InvertImg(p=.05),
                #A.RGBShift(p=.15),
                A.OneOf([
                    A.Solarize(p=.5, threshold_range=(0.8, 0.8)),
                    A.Solarize(p=.5, threshold_range=(0.9, 0.9)),
                ], p=.1),

                #A.Superpixels(p=.1, max_size=12, n_segments=(2, 8)),

                A.OneOf([
                    A.ToGray(p=.6),
                    A.ToSepia(p=.4),
                ], p=.1),

                A.CoarseDropout(p=.1),
                # A.HorizontalFlip(p=.1),
                # A.Morphological(p=.15, operation="erosion", scale=3),
                # A.Morphological(p=.15, operation="dilation", scale=3),
                A.PixelDropout(p=.1, per_channel=True),
                A.RandomRotate90(p=.1),
                # A.RandomSizedCrop(p=.1, min_max_height=(28, 28), size=(32, 32)),
                A.Rotate(p=.1),
                A.Transpose(p=.1),
            ])
        elif split == "test":
            self.transform = A.Compose(A.NoOp(p=1))

    def __len__(self) -> int:
        """Obtain length of the data.

        Returns:
            int: quantity of the data
        """
        return self.length

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, int]:
        """Get item of the dataset.

        Parameters:
            idx (int): index of the datapoint in the dataset

        Returns:
            a tuple containing an image and label
        """
        image_path = get_data_file_path(idx, 4, ".png")
        image = cv.imread(self.dataset_path / image_path)
        image = self.transform(image=image)["image"]
        image = image / 255
        image = image.reshape(3, 256, 256)

        label = self.dataset[idx]["label"]

        return image, label

class Covid19Dataset(Dataset):
    """Represents the Covid-19 Dataset.

    The Stanford Cars dataset contains 317 images of 3 classes. The
    class label usually consists of the make, model and year. Images have
    various sizes, some are png, but most are jpeg.

    The dataset is split like this:
        train = 251 images (111 = Covid, 70 = Viral Pneumonia, 70 = Normal)
        test = 66 images (26 = Covid, 20 = Viral Pneumonia, 20 = Normal)

    Source: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data

    Downloaded at: 2025/10/09/08/42
    """

    def __init__(
            self,
            dataset_path: str,
            ) -> None:
        """Construct a Covid19Dataset object.

        Constructs a Covid19Dataset object using the specified dataset path.

        Parameters:
            dataset_path (str): path to the datafile
        """
        self.dataset_path = Path(dataset_path)
        self.anno_file = pd.read_csv(self.dataset_path / "anno.csv")
        self.length = len(self.anno_file)
        self.labels = ["Covid", "Normal", "Viral Pneumonia"]

    def __len__(self) -> int:
        """Obtain length of the data.

        Returns:
            int: quantity of the data
        """
        return self.length

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, int]:
        """Get item of the dataset.

        Parameters:
            idx (int): index of the datapoint in the dataset

        Returns:
            a tuple containing an image and label
        """
        image_path = self.anno_file.at[idx, "path"]
        image = cv.imread(self.dataset_path / image_path)
        height, width, channels = image.shape
        # plt.imshow(image)
        # plt.show()
        image = image / 255
        image = image.reshape(channels, height, width)

        label = self.labels.index(str(Path(image_path).parent))

        # return image, label, image_path
        return image, label, idx


def get_data(
        batch_size: int,
        num_workers: int,
        data
        ) -> tuple[DataLoader, DataLoader]:
    """Obtain dataloaders.

    Parameters:
    batch_size (int):
        maximum number of images in one batch
    num_workers (int):
        number of concurrent processes
    data

    Returns:
    tuple consisting of two DataLoaders (train and test)
    """
    # Instantiate datasets
    train_data = instantiate(data.train)
    test_data = instantiate(data.test)

    # Create data loaders.
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )
    for x, y, _ in train_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of labels: {y.shape} {y.dtype}")
        break
    for x, y, _ in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of labels: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader
