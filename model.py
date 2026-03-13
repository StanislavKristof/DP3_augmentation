"""This file is responsible for creation of the model.

Taken from:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

from torch import nn, Tensor, zeros
from utils import (
    get_run_name,
    compute_rotation_matrix,
    compute_translation_matrix,
    compute_scaling_matrix,
    compute_bounded_sigmoid,
    compute_bounded_sigmoid_intersection
)
from pathlib import Path
import wandb
import torch
from icecream import ic
import timm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.clustering import MutualInfoScore, NormalizedMutualInfoScore
from typing import Tuple
import numpy as np
from torch.nn import functional
# import matplotlib.pyplot as plt

class AffineMixLoss(nn.Module):
    def __init__(
            self,
            device: str,
            ssim_data_range: Tuple[float, float] = (0.0, 1.0)
            ) -> None:
        super().__init__()
        self.ssim_data_range = ssim_data_range
        self.device = device
    
    def forward(
            self,
            augmented_imgs: Tensor,
            original_imgs: Tensor,
            rotation_radians: Tensor,
            translation_args: Tensor,
            scaling_args: Tensor,
            overfitting_score: float,
            rotation_weight: float = 0.5,
            translation_weight: float = 0.5,
            scaling_weight: float = 0.5,
            l2_weight: float = 0.5,
            ) -> Tuple[float, float, float, float, float]:
        # ic(augmented_imgs)
        # ic(augmented_imgs.shape)
        # ic(original_imgs)
        # ic(original_imgs.shape)
        # original_imgs_visual = torch.clone(original_imgs)
        # augmented_imgs_visual = torch.clone(augmented_imgs)
        # original_imgs_visual = torch.permute(original_imgs_visual, (0, 2, 3, 1))
        # augmented_imgs_visual = torch.permute(augmented_imgs_visual, (0, 2, 3, 1))

        augmented_imgs_clone = torch.clone(augmented_imgs)
        original_imgs_clone = torch.clone(original_imgs)

        # for augmented_img, original_img in zip(augmented_imgs_visual, original_imgs_visual):
        #     plt.imshow(original_img)
        #     plt.show()
        #     plt.imshow(augmented_img.detach().numpy())
        #     plt.show()

        rotation_val = torch.mean(torch.cos(rotation_radians) + 1)
        translation_val = torch.mean(2**0.5 - torch.sqrt(translation_args[0] ** 2 + translation_args[1] ** 2))
        scaling_val = torch.mean(torch.log(scaling_args + 1e-8))

        self.mis = MutualInfoScore().to(device=self.device)
        row_augmented_img =(torch.ravel(augmented_imgs_clone) * 255).type(dtype=torch.uint8)
        row_original_img = (torch.ravel(original_imgs_clone) * 255).type(dtype=torch.uint8)
        mis_val = self.mis(row_augmented_img, row_original_img)

        # self.l2 = nn.MSELoss().to(device=self.device)
        # l2_val = self.l2(original_imgs, augmented_imgs) + 1e-8
        # l2_val = torch.sqrt(l2_val)
        # l2_val = l2_val.type(torch.float32)

        final_loss = rotation_weight * rotation_val + translation_weight * translation_val + scaling_weight * scaling_val - overfitting_score * l2_weight * mis_val

        return final_loss, rotation_val, translation_val, scaling_val, mis_val


# https://docs.pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class SpatialTransformerNetwork(nn.Module):
    """Responsible for augmentation of the image (scaling, rotation and translation)."""

    def __init__(
            self,
            image_size: int,
            learning_rate: float,
            epochs: int,
            device: str,
            scaling_bounds: float,
            translation_bounds: float,
            rotation_bounds: float,
            padding_mode: str
            ) -> None:
        super().__init__()
        self.name = "affine_augmentator"
        self.scaling_lower_bound = scaling_bounds["min"]
        self.scaling_upper_bound = scaling_bounds["max"]
        self.translation_lower_bound = translation_bounds["min"]
        self.translation_upper_bound = translation_bounds["max"]
        self.rotation_lower_bound = rotation_bounds["min"]
        self.rotation_upper_bound = rotation_bounds["max"]
        self.padding_mode = padding_mode

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.rotation = nn.Sequential(
            nn.Linear(in_features=10 * 4 * 4, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1)
        )

        self.translation = nn.Sequential(
            nn.Linear(in_features=10 * 4 * 4, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, out_features=2)
            # nn.Tanh()
        )

        self.scaling = nn.Sequential(
            nn.Linear(in_features=10 * 4 * 4, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=2)
            # nn.Sigmoid()
        )

        # initialize the weights/bias for no transformation
        self.rotation[6].weight.data.zero_()
        self.rotation[6].bias.data.copy_(torch.tensor([0 + 1e-8], dtype=torch.float))

        self.translation[6].weight.data.zero_()
        self.translation[6].bias.data.copy_(torch.tensor([0 + 1e-4, 0+1e-4], dtype=torch.float))

        self.scaling[6].weight.data.zero_()
        self.scaling[6].bias.data.copy_(
            torch.tensor(
                2*[
                    compute_bounded_sigmoid_intersection(
                        intersection=1,
                        lower_bound=self.scaling_lower_bound,
                        upper_bound=self.scaling_upper_bound
                    )
                ], dtype=torch.float)
        )

        self.device = device
        self.loss_function = AffineMixLoss(device=self.device, ssim_data_range=(0.0, 1.0))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def stn(self, x):
        xs = self.localization(x)
        
        xs = xs.view(-1, 10 * 4 * 4)
        rotation_radians = self.rotation(xs)
        rotation_radians = compute_bounded_sigmoid(rotation_radians, lower_bound=self.rotation_lower_bound, upper_bound=self.rotation_upper_bound)
        translation_args = self.translation(xs)
        translation_args = compute_bounded_sigmoid(translation_args, lower_bound=self.translation_lower_bound, upper_bound=self.translation_upper_bound)
        scaling_args = self.scaling(xs)
        scaling_args = compute_bounded_sigmoid(scaling_args, lower_bound=self.scaling_lower_bound, upper_bound=self.scaling_upper_bound)
        x_batch = []

        for x_img, rotation_radian, translation_arg, scaling_arg in zip(x, rotation_radians, translation_args, scaling_args):
            rotation_matrix = compute_rotation_matrix(radians=rotation_radian, device=self.device)
            translation_matrix = compute_translation_matrix(t_x=translation_arg[0], t_y=translation_arg[1], device=self.device)
            scaling_matrix = compute_scaling_matrix(s_x=scaling_arg[0], s_y=scaling_arg[1], device=self.device)
            affine_matrix = translation_matrix @ scaling_matrix @ rotation_matrix
            theta = affine_matrix[:2]
            theta = theta.view(-1, 2, 3)
            x_img = x_img.unsqueeze(0)
            grid = functional.affine_grid(theta, x_img.size(), align_corners=False)
            x = functional.grid_sample(x_img, grid, align_corners=False, padding_mode=self.padding_mode) 
            x_batch.append(x.squeeze())

        x_batch = torch.stack(x_batch, dim=0).to(torch.float).to(device=self.device)
        return x_batch, rotation_radians, translation_args, scaling_args

    def forward(self, x):
        x = x.float()
        x, rotation_radians, translation_args, scaling_args = self.stn(x)
        return x, rotation_radians, translation_args, scaling_args

class MixLoss(nn.Module):
    def __init__(
            self,
            device: str,
            ssim_data_range: Tuple[float, float] = (0.0, 1.0)
            ) -> None:
        super().__init__()
        self.ssim_data_range = ssim_data_range
        self.device = device
    
    def forward(
            self,
            augmented_imgs: Tensor,
            original_imgs: Tensor,
            overfitting_score: float,
            ssim_weight: float = 0.5,
            l2_weight: float = 0.5,
            ) -> Tuple[float, float, float]:
        self.ssim = StructuralSimilarityIndexMeasure(data_range=self.ssim_data_range).to(device=self.device)
        self.l2 = nn.MSELoss().to(device=self.device)
        original_imgs = original_imgs.type(torch.float32)
        # for original_img, augmented_img in zip(original_imgs, augmented_imgs):
        #     original_img_visual = torch.clone(original_img).detach().cpu()
        #     augmented_img_visual = torch.clone(augmented_img).detach().cpu()
        #     original_img_visual = torch.permute(original_img_visual, (1, 2, 0))
        #     augmented_img_visual = torch.permute(augmented_img_visual, (1, 2, 0))
        #     
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     ax1.imshow(original_img_visual)
        #     ax2.imshow(augmented_img_visual)
        #     plt.show()
        l2_val = self.l2(original_imgs, augmented_imgs)
        l2_val = torch.sqrt(l2_val)
        l2_val = l2_val.type(torch.float32)

        ssim_val = self.ssim(augmented_imgs, original_imgs)
        final_loss = overfitting_score * ssim_weight * ssim_val + l2_weight * l2_val

        return final_loss, ssim_val, l2_val

class UNetAugmentatorTwo(nn.Module):
    """Responsible for augmentation of the image (hue shift)."""

    def __init__(
            self,
            image_size: int,
            learning_rate: float,
            # weight_decay: float,
            epochs: int,
            device: str
            ) -> None:
        super().__init__()
        self.name = "hue_augmentator"

        # 3, 32, 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 32, 32, 32
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32, 16, 16

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 64, 16, 16

        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64, 8, 8

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 128, 8, 8
    
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 64, 16, 16

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 64, 16, 16

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 32, 32, 32

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh() # this tanh right here is very important
        )
        self.device = device
        self.loss_function = MixLoss(device=self.device, ssim_data_range=(0.0, 1.0))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        original_x = self.conv1(x)
        after_down1_x = self.down1(original_x)
        after_down1_x = self.conv2(after_down1_x)
        after_down2_x = self.down2(after_down1_x)
        after_down2_x = self.conv3(after_down2_x)
        after_down2_x = self.up1(after_down2_x)
        after_down_sum = torch.cat([after_down1_x, after_down2_x], 1)
        after_down_sum = self.conv4(after_down_sum)
        after_down_sum = self.up2(after_down_sum)
        original_after_down_sum = torch.cat([original_x, after_down_sum], 1)
        x = self.conv5(original_after_down_sum)
        x = self.conv6(x)
        return x

class UNetAugmentatorSmall(nn.Module):
    """Responsible for augmentation of the image using UNet (hue shift)."""
    def __init__(
            self,
            image_size: int,
            learning_rate: float,
            # weight_decay: float,
            epochs: int,
            device: str
            ) -> None:
        super().__init__()
        self.name = "hue_augmentator"

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
    
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.loss_function = MixLoss(ssim_data_range=(0.0, 1.0), device=device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        original_x = self.conv1(x)
        down_x = self.down1(original_x)
        down_x = self.conv2(down_x)
        down_x = self.up1(down_x)
        down_original_sum = torch.cat([down_x, original_x], 1)
        x = self.conv3(down_original_sum)
        x = self.conv4(x)
        return x

class ClassicAugmentator(nn.Module):
    """Responsible for augmentation of the image."""

    def __init__(
            self,
            image_size: int,
            learning_rate: float,
            # weight_decay: float,
            epochs: int
            ) -> None:
        super().__init__()

        self.Conv = nn.Sequential(
            # 3, 32, 32
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),

            # 4, 16, 16
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),

            # 8, 8, 8
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),

            # 16, 4, 4
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),

            # 32, 2, 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(),

            # 16, 4. 4
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=1, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout(),

            # 8, 8, 8
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=1, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.Dropout(),

            # 4, 16, 16
            nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # 3, 32, 32

            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            # 3, 32, 32
        )
        self.loss_function = MixLoss(ssim_data_range=(0.0, 1.0))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=epochs/256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.Conv(x)
        return x

class SVHNClassifier(nn.Module):
    """Custom model based on model used in DP_old/SVHN_with_train_mat.2.

    This is the model described in DP1 Interim.
    """
    def __init__(
            self,
            learning_rate: float
            ) -> None:
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=128 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10)
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            # weight_decay=weight_decay
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        """Compute output based on input x.

        Parameters:
        x (torch.Tensor):
            input of the network

        Returns:
        output of the network
        """
        x = x.float()
        logits = self.Conv.forward(x)
        return logits




class CIFAR10Classifier(nn.Module):
    """Custom model based on model used in DP_old/CIFAR10_fixed_small_inputs.6.

    This is the model described in DP1 Interim.
    """
    def __init__(
            self,
            learning_rate: float
            ) -> None:
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(in_features=128 * 8 * 8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            # weight_decay=weight_decay
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        """Compute output based on input x.

        Parameters:
        x (torch.Tensor):
            input of the network

        Returns:
        output of the network
        """
        x = x.float()
        logits = self.Conv.forward(x)
        return logits





        
class NeuralNetwork(nn.Module):
    """The actual classifier."""

    def __init__(
            self,
            feature_extractor: str,
            number_of_classes: int,
            image_size: int,
            learning_rate: float,
            # weight_decay: float,
            epochs: int
            ) -> None:
        """Create model's architecture.

        Parameters:
        architecture (str):
            architecture of the created model described using Python code
        feature_extractor (str):
            selected pretrained feature extractor from Pytorch Image Models
            (TIMM)
        """
        super().__init__()

        self.feature_extractor = timm.create_model(
            feature_extractor,
            pretrained=False,
            num_classes=number_of_classes
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            # weight_decay=weight_decay
        )
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=epochs/256)

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        """Compute output based on input x.

        Parameters:
        x (torch.Tensor):
            input of the network

        Returns:
        output of the network
        """
        x = x.float()
        logits = self.feature_extractor.forward(x)
        return logits


def get_models(
        device: str,
        feature_extractor: str,
        number_of_classes: int,
        image_size: int,
        primary_learning_rate: float,
        # primary_weight_decay: float,
        hue_augmentator_learning_rate: float,
        affine_augmentator_learning_rate: float,
        # augmentator_weight_decay: float,
        epochs: int,
        scaling_bounds: float,
        translation_bounds: float,
        rotation_bounds: float,
        padding_mode: str
        ) -> NeuralNetwork:
    """Create model and move it into specified device.

    Parameters:
    device (str):
        device, where model will be moved
    architecture (str):
        architecture of the created model described using Python code
    feature_extractor (str):
        selected pretrained feature extractor from Pytorch Image Models (TIMM)

    Returns:
    created model moved into the specified device
    """
    if feature_extractor == "svhn":
        feature_extractor = SVHNClassifier(
            learning_rate=primary_learning_rate,
        ).to(device)
    elif feature_extractor == "cifar10":
        feature_extractor = CIFAR10Classifier(
            learning_rate=primary_learning_rate,
        ).to(device)
    else:
        feature_extractor = NeuralNetwork(
            feature_extractor=feature_extractor,
            number_of_classes=number_of_classes,
            image_size=image_size,
            learning_rate=primary_learning_rate,
            # weight_decay=primary_weight_decay,
            epochs=epochs
        ).to(device)
        
    return (
        feature_extractor,
        UNetAugmentatorSmall(
            image_size=image_size,
            learning_rate=hue_augmentator_learning_rate,
            # weight_decay=augmentator_weight_decay,
            epochs=epochs,
            device=device
        ).to(device),
        SpatialTransformerNetwork(
            image_size=image_size,
            learning_rate=affine_augmentator_learning_rate,
            # weight_decay=augmentator_weight_decay,
            epochs=epochs,
            device=device,
            scaling_bounds=scaling_bounds,
            translation_bounds=translation_bounds,
            rotation_bounds=rotation_bounds,
            padding_mode=padding_mode
        ).to(device)
    )


def save_model(model: nn.Module, is_augment = False) -> None:
    """Save model to models/<model_name>/<model_version>.pt.

    Parameters:
    model (nn.Module):
        model to save
    """
    name, version = get_run_name(wandb.run)
    dir_name = Path(f"models/{name}")
    if is_augment:
        file_name = Path(f"{model.name}_{version}.pt")
    else:
        file_name = Path(f"{version}.pt")
    try:
        dir_name.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    model_location = dir_name / file_name
    torch.save(model.state_dict(), model_location)
    print(f"Saved PyTorch Model State to {model_location}")
