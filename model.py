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
from typing import Tuple, List
import numpy as np
from torch.nn import functional
import math
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

class UNetAugmentator256(nn.Module):
    """Responsible for augmentation of the image using UNet (hue shift) for 256, 256 images."""
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

        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.loss_function = MixLoss(ssim_data_range=(0.0, 1.0), device=device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W
        x = x.float()

        # B, 3, 256, 256
        after_conv1 = self.conv1(x)

        # B, 64, 256, 256
        after_down1 = self.down1(after_conv1)

        # B, 64, 128, 128
        after_conv2 = self.conv2(after_down1)

        # B, 128, 128, 128
        after_down2 = self.down2(after_conv2)

        # B, 128, 64, 64
        after_conv3 = self.conv3(after_down2)

        # B, 256, 64, 64
        after_up1 = self.up1(after_conv3)

        # B, 128, 128, 128
        after_cat1 = torch.cat([after_conv2, after_up1], 1)

        # B, 256, 128, 128
        after_conv4 = self.conv4(after_cat1)

        # B, 128, 128, 128
        after_up2 = self.up2(after_conv4)

        # B, 64, 256, 256
        after_cat2 = torch.cat([after_conv1, after_up2], 1)

        # B, 128, 256, 256
        after_conv5 = self.conv5(after_cat2)

        # B, 64, 256, 256
        after_conv6 = self.conv6(after_conv5)

        # B, 3, 256, 256
        return after_conv6

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


# class CIFAR10Classifier(nn.Module):
#     """New classifier for CIFAR10. 
#     
#     As proposed by https://github.com/shashwat-shahi in
#     https://github.com/shashwat-shahi/CIFAR-10-Image-Classification
#     """
#     def __init__(
#             self,
#             learning_rate: float
#             ) -> None:
#         super().__init__()
# 
#         self.Conv = nn.Sequential(
#             # conv block 1
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.2),
# 
#             # conv block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.3),
#             # conv block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.4),
# 
#             # fully connected layers
#             nn.Flatten(),
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 10),
#             nn.ReLU()
#             
#             # original keras architecture
#             # first conv block
#             # layers.Input(shape=(32, 32, 3)),
#             # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.MaxPooling2D((2, 2)),
#             # layers.Dropout(0.2),
#             # # second conv block
#             # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.MaxPooling2D((2, 2)),
#             # layers.Dropout(0.3),
#             # # second conv block
#             # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#             # layers.BatchNormalization(),
#             # layers.MaxPooling2D((2, 2)),
#             # layers.Dropout(0.4),
#             # # fully connected 
#             # layers.Flatten(),
#             # layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#             # layers.BatchNormalization(),
#             # layers.Dropout(0.5),
#             # layers.Dense(10, activation='softmax')
#         )
#         self.optimizer = torch.optim.Adam(
#             self.parameters(),
#             lr=learning_rate,
#             eps=1e-08, # originally written in keras which uses eps=1-08e by default
#             weight_decay=0.001 # originally used regularizer l2 with 0.001 at the penultimate dense (linear) layer
#         )
#         self.loss_function = nn.CrossEntropyLoss()
# 
#     def forward(
#             self,
#             x: torch.Tensor
#             ) -> torch.Tensor:
#         """Compute output based on input x.
# 
#         Parameters:
#         x (torch.Tensor):
#             input of the network
# 
#         Returns:
#         output of the network
#         """
#         x = x.float()
#         logits = self.Conv.forward(x)
#         return logits

# class CIFAR10Classifier(nn.Module):
#     """Custom model based on model used in DP_old/CIFAR10_fixed_small_inputs.6.
# 
#     This is the model described in DP1 Interim.
#     """
#     def __init__(
#             self,
#             learning_rate: float
#             ) -> None:
#         super().__init__()
# 
# vstupna conv, resblok 256 kanalmi
#         self.Conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             
#             nn.Flatten(),
#             nn.Linear(in_features=128 * 8 * 8, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=10),
#             nn.ReLU()
#         )
#         self.optimizer = torch.optim.Adam(
#             self.parameters(),
#             lr=learning_rate,
#             # weight_decay=weight_decay
#         )
#         self.loss_function = nn.CrossEntropyLoss()
# 
#     def forward(
#             self,
#             x: torch.Tensor
#             ) -> torch.Tensor:
#         """Compute output based on input x.
# 
#         Parameters:
#         x (torch.Tensor):
#             input of the network
# 
#         Returns:
#         output of the network
#         """
#         x = x.float()
#         logits = self.Conv.forward(x)
#         return logits
"""Resnet for CIFAR10

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
    author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    title     = {Deep Residual Learning for Image Recognition},
    booktitle = {{CVPR}},
    pages     = {770--778},
    publisher = {{IEEE} Computer Society},
    year      = {2016}
}

"""
# __all__ = [
#     'resnet20_cifar_fp4', 'resnet32_cifar_fp4', 'resnet44_cifar_fp4',
#     'resnet56_cifar_fp4', 'resnet20_cifar_sfp4', 'resnet32_cifar_sfp4',
#     'resnet44_cifar_sfp4', 'resnet56_cifar_sfp4', 'resnet20_cifar',
#     'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar'
# ]

def conv3x3(conv2d, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(conv, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(conv, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        # self.residual_eltwiseadd = EltwiseAdd() # stačí plusko

        # self.scaling_a = ScalingLayer(planes)
        # self.scaling_b = ScalingLayer(planes)
        # if downsample is not None:
        #     self.scaling_a = None
        #     self.scaling_b = None

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # else:
        #     residual = self.scaling_a(residual)
        #     out = self.scaling_b(out)

        # out = self.residual_eltwiseadd(residual, out)
        out = residual + out
        out = self.relu2(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(
            self,
            conv: nn.modules.conv,
            block: nn.Module,
            layers: List[int],
            learning_rate: float,
            weight_decay: float,
            num_classes: int = 10
            ) -> None:
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
        self.features = [
            nn.Conv2d(3, self.inplanes, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(), # nn.GELU(),
            self._make_layer(self.layer_gates[0], conv, block, 16, layers[0]),
            self._make_layer(self.layer_gates[1], conv, block, 32, layers[1], stride=2),
            self._make_layer(self.layer_gates[2], conv, block, 64, layers[2], stride=2),
            nn.AvgPool2d(8, stride=1),
            nn.Flatten()
        ]

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(*[
            nn.Linear(64 * block.expansion, num_classes)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        #     eps=1e-08, # originally written in keras which uses eps=1-08e by default
        #     weight_decay=0.001 # originally used regularizer l2 with 0.001 at the penultimate dense (linear) layer
        )

    def _make_layer(self, layer_gates, conv, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(conv, layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(conv, layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        phi = self.features(x)
        x = self.classifier(phi)

        return x


class NeuralNetwork(nn.Module):
    """The actual classifier."""

    def __init__(
            self,
            feature_extractor: str,
            number_of_classes: int,
            image_size: int,
            learning_rate: float,
            weight_decay: float,
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
            weight_decay=weight_decay
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
        hue_augmentator: str,
        number_of_classes: int,
        image_size: int,
        primary_learning_rate: float,
        primary_weight_decay: float,
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
    elif feature_extractor == "resnet20_cifar":
        feature_extractor = ResNetCifar(
            conv=nn.Conv2d,
            block=BasicBlock,
            layers=[3, 3, 3],
            learning_rate=primary_learning_rate,
            weight_decay=primary_weight_decay,
            num_classes=number_of_classes
            ).to(device)
    elif feature_extractor == "resnet32_cifar":
        feature_extractor = ResNetCifar(
            conv=nn.Conv2d,
            block=BasicBlock,
            layers=[5, 5, 5],
            learning_rate=primary_learning_rate,
            weight_decay=primary_weight_decay,
            num_classes=number_of_classes
            ).to(device)
    elif feature_extractor == "resnet44_cifar":
        feature_extractor = ResNetCifar(
            conv=nn.Conv2d,
            block=BasicBlock,
            layers=[7, 7, 7],
            learning_rate=primary_learning_rate,
            weight_decay=primary_weight_decay,
            num_classes=number_of_classes
            ).to(device)
    elif feature_extractor == "resnet56_cifar":
        feature_extractor = ResNetCifar(
            conv=nn.Conv2d,
            block=BasicBlock,
            layers=[9, 9, 9],
            learning_rate=primary_learning_rate,
            weight_decay=primary_weight_decay,
            num_classes=number_of_classes
            ).to(device)
    else:
        feature_extractor = NeuralNetwork(
            feature_extractor=feature_extractor,
            number_of_classes=number_of_classes,
            image_size=image_size,
            learning_rate=primary_learning_rate,
            weight_decay=primary_weight_decay,
            epochs=epochs
        ).to(device)

    if hue_augmentator == "hue_augmentator_256":
        hue_augmentator = UNetAugmentator256(
            image_size=image_size,
            learning_rate=hue_augmentator_learning_rate,
            epochs=epochs,
            device=device
        ).to(device)
    else:
        hue_augmentator = UNetAugmentatorSmall(
            image_size=image_size,
            learning_rate=hue_augmentator_learning_rate,
            epochs=epochs,
            device=device
        ).to(device)

    return (
        feature_extractor,
        hue_augmentator,
        SpatialTransformerNetwork(
            image_size=image_size,
            learning_rate=affine_augmentator_learning_rate,
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
