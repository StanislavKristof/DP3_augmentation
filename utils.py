"""A file used as utils."""

from ruamel.yaml import YAML
import re
import numpy as np
from math import (
    prod,
    sqrt,
    log
)
from pathlib import Path
import os
from typing import Union, Dict
import wandb
from icecream import ic
import pystache
from deprecated import deprecated
from torch import nn, Tensor
import torch


def update_version(
        config_file: Union[Path, str]
        ) -> None:
    """Update version number in the config file.

    Parameters
    ----------
    config_file : Union[Path, str]
        config file whose version is to be updated
    """
    with open(config_file) as f:
        content = f.read()

    yaml = YAML(typ="safe", pure=True)
    version = yaml.load(content)["config"]["train_info"]["version"]

    content = re.sub(
        rf'version\s*:\s*{version}', f'version: {version + 1}', content
        )

    with open(config_file, "w") as f:
        f.write(content)


def get_run_name(
        wandb_run: Union[wandb.run, None]
        ) -> tuple[str, str]:
    """Split run name into name and version.

    Parameters:
    name (wandb_sdk.wandb_run.Run / None):
        wandb.run object

    Returns:
    a tuple consisting of name and version
    """
    if wandb_run is None:
        return "test", "0"
    dot_index = wandb_run.name.rfind(".")
    return wandb_run.name[:dot_index], wandb_run.name[dot_index + 1:]


def get_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two 2d points - a and b.

    Parameters:
    a (np.ndarray):
        a numpy array containing two numbers - x coord and y coord
    b (np.ndarray):
        a numpy array containing two numbers - x coord and y coord
    """
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_data_file_path(
        id: int,
        number_of_digits: int,
        suffix: str
        ) -> Path:
    """Return a new file path for data (e.g. image, annotation file).

    The path is based on the provided ID.

    Parameters
    ----------
    id : int
        The ID of the datapoint.
    number_of_digits : int
        Number of digits in the maximal ID.
    suffix : str
        A suffix to be added.

    Returns
    -------
    Path
        The path to the data file.
    """
    id_str = str(id).zfill(number_of_digits)
    return Path(id_str[:-3] + "XXX") / Path(id_str).with_suffix(suffix)


def create_parent_directory_if_not_exists(file_path: Path) -> None:
    """Create a parent directory of the file in the path.

    If it does not already exist, creates a parent directory of the
    file in the path.

    Parameters:
    file_path (Path):
        file whose parent directory will be created
    """
    parent_directory = file_path.parent
    if not parent_directory.exists():
        os.mkdir(parent_directory)


def absolute_to_relative(id: int, width: int, height: int) -> None:
    """Change all bboxes in bboxes dir from absolute to relative.

    Parameters:
    id (int):
        id of an image
    width (int):
        width of an image
    height (int):
        height of an image
    """
    bboxes_dir = Path("data/cars/bboxes")
    file_name = get_new_file_name(id).with_suffix(".npy")
    bboxes = np.load(bboxes_dir / file_name)
    bboxes = bboxes.astype("float64")
    bboxes[:, :, 0] = bboxes[:, :, 0] / width
    bboxes[:, :, 1] = bboxes[:, :, 1] / height
    new_bboxes_dir = Path("data/cars/relativebboxes")
    create_parent_directory_if_not_exists(new_bboxes_dir / file_name)
    np.save(new_bboxes_dir / file_name, bboxes)


def prepare_config_hydra_yaml(
        config_path: Union[Path, str],
        config_template_path: Union[Path, str],
        config_hydra_path: Union[Path, str]
        ) -> None:
    """Prepare the config hydra file.

    Prepare the config hydra file by inserting the data from config file to
    the config template.

    Parameters:
    config_path (Path / str):
        path to the primary config file (edited manually)
    config_template_path (Path / str):
        path to the config template
    config_hydra_path (Path / str):
        path to the config hydra (to be constructed)
    """
    with open(config_path) as config:
        config_content = config.read()

    yaml = YAML(typ="safe", pure=True)
    config_content = yaml.load(config_content)

    with open(config_template_path) as config_template:
        config_template_content = config_template.read()

    config_hydra_content = pystache.render(
        config_template_content,
        config_content
    )
    with open(config_hydra_path, "w") as config_hydra:
        config_hydra.write(config_hydra_content)


def compute_recommended_learning_rate_range(
        shape: tuple[int],
        ) -> tuple[float]:
    """Compute the recommended learning rate range based on shape of the input.

    The recommended learning rate range is 0.1 <= n * lr <= 1

    Parameters
    ----------
    shape : tuple[int]
        The shape of the input

    Returns
    -------
    tuple[float]
        Return a tuple containing the boundaries of the learning rate.
    """
    n = prod(shape)
    lower_boundary = 0.1 / n
    upper_boundary = 1 / n
    return lower_boundary, upper_boundary


def is_in_colab() -> bool:
    """Find out whether the code is executed in the Google Colab Environment.

    Returns
    -------
    bool
        Return whether the code is executed in the Google Colab Environment.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def add_colab_path(
    *args: Union[Path, str],
    to_be_added : Path = Path("drive/MyDrive/folly")
    ) -> Union[Path, tuple[Path]]:
    result = tuple([to_be_added / Path(arg) for arg in args])
    if len(result) == 1:
        return result[0]
    return result


def log_code_from() -> Path:
    if is_in_colab():
        return Path("/content/drive/MyDrive/folly")
    return Path(".")


def track_shape(shape: tuple, conv_network: nn.Sequential):
    tensor = torch.rand(shape)
    for layer in conv_network:
        print(f"before {layer}: {tensor.shape}")
        tensor = layer(tensor)
        print(f"after {layer}: {tensor.shape}")

def save_images(
        tensor_dict: Dict[str, Tensor],
        # tensor: List[Tensor],
        # augmentator_name: List[str],
        wandb,
        epoch: int
        ) -> None:
    """Save images tensor to images/<model_name>/<model_version>/<epoch>/<image_path>

    Parameters:
    tensor (Tensor):
        tensor of images to be save
    """
    name, version = get_run_name(wandb.run)
    # dir_name = Path(f"images/{name}/{augmentator_name}/{version}")
    dir_name = Path(f"images/{name}/{version}")
    try:
        dir_name.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    tensor_dict = {
        augmentation_name: (tensor * 255).astype(np.uint8)
        for augmentation_name, tensor in tensor_dict.items()
    }
    # tensor = tensor * 255
    # tensor = tensor.astype(np.uint8)
    np.savez_compressed(dir_name / str(epoch), **tensor_dict)#arr=tensor)

def get_device(device: str) -> str:
    if device is None:
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return device


# @deprecated(reason="""
#     separate images took too much space, saving whole batches
#     as .npy is to be used from now on""")
# def save_image(
#         tensor: Tensor,
#         image_path: Path,
#         wandb,
#         # model_name: str,
#         # model_version,
#         epoch: int
#         ) -> None:
#     """Save image to images/<model_name>/<model_version>/<epoch>/<image_path>
# 
#     Parameters:
#     tensor (Tensor):
#         tensor to be converted to image to save
#     """
#     image_path = Path(image_path)
#     name, version = get_run_name(wandb.run)
#     dir_name = Path(f"images/{name}/{version}/{epoch}/{image_path.parent}")
#     try:
#         dir_name.mkdir(parents=True, exist_ok=False)
#     except FileExistsError:
#         pass
# 
#     tensor_copy = torch.clone(tensor)
#     image = tensor_copy.detach().cpu().numpy()
#     c, h, w = image.shape
#     image = image.reshape(h, w, c) # COVID
#     # image = np.transpose(image, axes=(1, 2, 0)) # SVHN
#     image = image * 255
#     image = image.astype(np.uint8)
#     cv.imwrite(dir_name / image_path.name, image)

def compute_translation_matrix(t_x: float, t_y: float, device: str) -> torch.Tensor:
    first_row = torch.cat(tensors=(torch.tensor([1., 0.], requires_grad=True, device=device), t_x.view(1))).view(3)
    second_row = torch.cat(tensors=(torch.tensor([0., 1.], requires_grad=True, device=device), t_y.view(1))).view(3)
    third_row = torch.tensor([0., 0., 1.], device=device)
    result = torch.stack(tensors=(first_row, second_row, third_row))
    return result

def compute_rotation_matrix(radians: float | torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(radians, float):
        radians = torch.tensor([radians], requires_grad=True)
    first_row = torch.stack(tensors=(torch.cos(radians), -torch.sin(radians), torch.tensor([0.], requires_grad=True, device=device))).view(3)
    second_row = torch.stack(tensors=(torch.sin(radians), torch.cos(radians), torch.tensor([0.], requires_grad=True, device=device))).view(3)
    third_row = torch.tensor([0., 0., 1.], device=device)
    result = torch.stack(tensors=(first_row, second_row, third_row))
    return result

def compute_scaling_matrix(s_x: float, s_y: float, device: str) -> torch.Tensor:
    first_row = torch.cat(tensors=(s_x.view(1), torch.tensor([0., 0.], requires_grad=True, device=device))).view(3)
    second_row = torch.cat(tensors=(torch.tensor([0.], device=device), s_y.view(1), torch.tensor([0.], device=device))).view(3)
    third_row = torch.tensor([0., 0., 1.], device=device)
    result = torch.stack(tensors=(first_row, second_row, third_row))
    return result

def compute_bounded_sigmoid(
        tensor: Tensor,
        lower_bound: float,
        upper_bound: float
        ) -> Tensor:
    return lower_bound + (upper_bound - lower_bound) * torch.nn.functional.sigmoid(tensor)

def compute_bounded_sigmoid_intersection(
        intersection: float,
        lower_bound: float,
        upper_bound: float
        ) -> Tensor:
    return log((lower_bound - intersection) / (intersection - upper_bound))