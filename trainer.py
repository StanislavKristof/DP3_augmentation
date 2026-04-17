"""This file is responsible for training the model."""

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Literal
import wandb
from metrics import MetricsCollection
from model import save_model
from icecream import ic
from torchmetrics.clustering import MutualInfoScore
import numpy as np
from utils import save_images

def train_epoch(
        dataloader: DataLoader,
        primary_model: nn.Module,
        hue_augmentator: nn.Module,
        affine_augmentator: nn.Module,
        device: Literal["cuda", "mps", "cpu"],
        ssim_weight: float,
        rotation_weight: float,
        translation_weight: float,
        scaling_weight: float,
        hue_l2_weight: float,
        affine_l2_weight: float,
        wandb,
        epoch: int,
        to_be_augmented: bool,
        to_save_augmented: bool,
        loss_fn: nn.modules.loss._Loss,
        overfitting_score: float
        ) -> MetricsCollection:
    """Train the model for an epoch.

    Parameters:
    dataloader (DataLoader):
        contains the training data
    model (nn.Module):
        model to be trained
    loss_fn (nn.modules.loss._Loss): 
        the loss function of choice
    optimizer (torch.optim):
        the optimizer of choice
    scheduler (torch.optim.lr_scheduler):
        the scheduler of choice
    device (Literal["cuda", "mps", "cpu"]):
        device where the training happens

    Returns:
    dict of metrics of the epoch (MetricsCollection)
    """
    num_batches = len(dataloader)

    # Create metrics
    metric_collection = MetricsCollection("train")
    metric_collection.add_metric("primary_loss")
    metric_collection.add_metric("accuracy")
    metric_collection.add_metric("top5_accuracy")
    primary_model.train()
    if to_be_augmented or True:
        metric_collection.add_metric("hue_augmentator_loss")
        metric_collection.add_metric("ssim_loss")
        metric_collection.add_metric("affine_augmentator_loss")
        metric_collection.add_metric("hue_L2")
        metric_collection.add_metric("affine_L2")
        metric_collection.add_metric("rotation_loss")
        metric_collection.add_metric("translation_loss")
        metric_collection.add_metric("scaling_loss")
    hue_augmentator.train()
    affine_augmentator.train()

    if to_be_augmented:
        for x, y, _ in dataloader:
            to_save_shape = (len(dataloader.dataset), x.shape[1], x.shape[2], x.shape[3])
            # to_save_hue = np.zeros(to_save_shape)
            # to_save_affine = np.zeros(to_save_shape)
            break
        to_save = {
            "hue": np.zeros(to_save_shape),
            "affine": np.zeros(to_save_shape),
        }
    progress_bar = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc="Training"
        )

    for i, batch in progress_bar:
        x, y = batch[0], batch[1]
        if len(batch) == 3:
            indices = batch[2]


        x, y = x.to(device), y.to(device)
        
        x = x.float()
        hue_aug_loss = affine_aug_loss = 0

        if to_be_augmented:
            x_cpu = torch.clone(x)
            x_hue_augmentation = hue_augmentator(x)
            x_hue_augmented = x + x_hue_augmentation
            x_hat, rotation_radians, translation_args, scaling_args = affine_augmentator(x_hue_augmented)

            x_cpu = torch.clone(x).cpu()
            for i, (x_hue_augmented_image, x_hat_image) in enumerate(zip(x_hue_augmented, x_hat)):
                # save affine
                x_hat_copy = torch.clone(x_hat_image).cpu()
                x_hat_copy = x_hat_copy.detach().cpu().numpy()
                if to_be_augmented:
                    to_save["affine"][indices[i]] = x_hat_copy

                # save hue
                x_hue_copy = torch.clone(x_hue_augmented_image).cpu()
                x_hue_copy = x_hue_copy.detach().cpu().numpy()
                if to_be_augmented:
                    to_save["hue"][indices[i]] = x_hue_copy

            hue_aug_loss, ssim_loss, hue_l2_loss = hue_augmentator.loss_function(
                augmented_imgs=x_hue_augmented,
                original_imgs=x,
                overfitting_score=overfitting_score,
                ssim_weight=ssim_weight,
                l2_weight=hue_l2_weight
            )
            affine_aug_loss, rotation_loss, translation_loss, scaling_loss, affine_l2_loss = affine_augmentator.loss_function(
                x_hat,
                x,
                rotation_radians=rotation_radians,
                rotation_weight=rotation_weight,
                translation_args=translation_args,
                translation_weight=translation_weight,
                scaling_args=scaling_args,
                scaling_weight=scaling_weight,
                l2_weight=affine_l2_weight,
                overfitting_score=overfitting_score
            )

            metric_collection["hue_augmentator_loss"].add(hue_aug_loss.item())
            metric_collection["ssim_loss"].add(ssim_loss.item())
            metric_collection["hue_L2"].add(hue_l2_loss.item())
            metric_collection["affine_augmentator_loss"].add(affine_aug_loss.item())
            metric_collection["affine_L2"].add(affine_l2_loss.item())
            metric_collection["rotation_loss"].add(rotation_loss.item())
            metric_collection["translation_loss"].add(translation_loss.item())
            metric_collection["scaling_loss"].add(scaling_loss.item())
        else:
            x_hat = x
            for i, x_hat_image in enumerate(x_hat):
                # save affine
                if to_save_augmented:
                    x_hat_copy = torch.clone(x_hat_image).cpu()
                    x_hat_copy = x_hat_copy.detach().cpu().numpy()
                    to_save["affine"][indices[i]] = x_hat_copy
                    to_save["hue"][indices[i]] = x_hat_copy
            metric_collection["hue_augmentator_loss"].add(0)
            metric_collection["ssim_loss"].add(0)
            metric_collection["hue_L2"].add(0)
            metric_collection["affine_augmentator_loss"].add(0)
            metric_collection["affine_L2"].add(0)
            metric_collection["rotation_loss"].add(0)
            metric_collection["translation_loss"].add(0)
            metric_collection["scaling_loss"].add(0)

        y_hat = primary_model(x_hat)

        # compute prediction error
        primary_loss = primary_model.loss_function(y_hat, y)
        metric_collection["primary_loss"].add(primary_loss.item())

        # backpropagation
        total_loss = hue_aug_loss + affine_aug_loss + primary_loss
        total_loss.backward()

        # resetting gradients so they do not accumulate
        hue_augmentator.optimizer.step()
        hue_augmentator.optimizer.zero_grad()
        affine_augmentator.optimizer.step()
        affine_augmentator.optimizer.zero_grad()
        primary_model.optimizer.step()
        primary_model.optimizer.zero_grad()

        # statistics
        # top 1 accuracy
        correct = int(
            (y_hat.argmax(dim=1) == y).type(torch.float).sum().item()
            )
        metric_collection["accuracy"].add(correct / y.shape[0])

        # top 5 accuracy
        values, indices = y_hat.topk(5)
        # ic(indices)
        # ic(type(indices))
        # ic(indices.shape)
        # ic(y.unsqueeze(1).shape)
        # ic(y.unsqueeze(1) == indices)
        matches = (y.unsqueeze(1) == indices).any(dim=1)
        # ic((y.unsqueeze(1) == indices).any(dim=1))
        # ic((y.unsqueeze(1) == indices).any(dim=1).sum().item())
        correct = int(matches.sum().item())
        # ic(correct)
        metric_collection["top5_accuracy"].add(correct / y.shape[0])

        progress_bar.set_postfix(metric_collection.get_dict("print"))
    if to_be_augmented and to_save_augmented:
        save_images(to_save, wandb, epoch)
    return (
        metric_collection.get_dict("wandb"),
        metric_collection["primary_loss"].get_average(),
        metric_collection["accuracy"].get_average(),
        metric_collection["hue_augmentator_loss"].get_average(),
        metric_collection["ssim_loss"].get_average(),
        metric_collection["hue_L2"].get_average(),
        metric_collection["affine_augmentator_loss"].get_average(),
        metric_collection["affine_L2"].get_average(),
        metric_collection["rotation_loss"].get_average(),
        metric_collection["translation_loss"].get_average(),
        metric_collection["scaling_loss"].get_average()
    )

def val_epoch(
        dataloader: DataLoader,
        primary_model: nn.Module,
        device: Literal["cuda", "mps", "cpu"]
        ) -> MetricsCollection:
    """Validate the model for an epoch.

    Parameters:
    dataloader (DataLoader):
        contains the training data
    model (nn.Module):
        model to be trained
    loss_fn (nn.modules.loss._Loss): 
        the loss function of choice
    device (Literal["cuda", "mps", "cpu"]):
        device where the training happens
    """
    num_batches = len(dataloader)

    # Create metrics
    metric_collection = MetricsCollection("val")
    metric_collection.add_metric("loss")
    metric_collection.add_metric("accuracy")
    metric_collection.add_metric("top5_accuracy")
    primary_model.eval()

    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader),
            total=num_batches,
            desc="Validation"
            )
        for i, batch in progress_bar:
            x, y  = batch[0], batch[1]

            x, y = x.to(device), y.to(device)

            # Compute prediction error
            y_hat = primary_model(x)
            loss = primary_model.loss_function(y_hat, y)
            metric_collection["loss"].add(loss.item())

            # Statistics
            # top 1 accuracy
            correct = int(
                (y_hat.argmax(dim=1) == y).type(torch.float).sum().item()
                )
            metric_collection["accuracy"].add(correct / y.shape[0])

            # top 5 accuracy
            values, indices = y_hat.topk(5)
            # ic(indices)
            # ic(type(indices))
            # ic(indices.shape)
            # ic(y.unsqueeze(1).shape)
            # ic(y.unsqueeze(1) == indices)
            matches = (y.unsqueeze(1) == indices).any(dim=1)
            # ic((y.unsqueeze(1) == indices).any(dim=1))
            # ic((y.unsqueeze(1) == indices).any(dim=1).sum().item())
            correct = int(matches.sum().item())
            # ic(correct)
            metric_collection["top5_accuracy"].add(correct / y.shape[0])

            progress_bar.set_postfix(metric_collection.get_dict("print"))
        return metric_collection.get_dict("wandb"), metric_collection["accuracy"].get_average(),


def train_models(
        epochs: int,
        # learning_rate: float,
        # weight_decay: float,
        primary_model,
        hue_augmentator,
        affine_augmentator,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device,
        number_of_classes: int,
        image_size,
        ssim_weight: float,
        rotation_weight: float,
        translation_weight: float,
        scaling_weight: float,
        hue_l2_weight: float,
        affine_l2_weight: float,
        wandb,
        to_be_augmented: bool,
        to_save_augmented: bool
        ) -> None:
    """Train the models for a number of epochs.

    Parameters:
    epochs (int): number of epoch for which the model will be trained
    learning_rate (f * t): learning rate used during the training
    weight_decay (float): weight decay used during the training
    primary_model : the classifier
    train_dataloader (DataLoader): contains training data
    val_dataloader (DataLoader): contains validation data
    device
    """

    # optimizer AdamW
    # optimizer = torch.optim.Adam(primary_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs/256)
    loss_fn = nn.CrossEntropyLoss()
    
    # there is no overfitting at the beggining of training
    overfitting_score = 0
    primary_losses = []
    train_accuracies = []
    val_accuracies = []
    hue_augmentator_losses = []
    ssim_losses = []
    hue_L2_losses = []
    affine_augmentator_losses = []
    affine_L2_losses = []
    rotation_losses = []
    translation_losses = []
    scaling_losses = []
    for epoch in range(1, epochs+1):

        # Training and validation
        print(f"EPOCH: {epoch}")
        train_dict, primary_loss, train_accuracy, hue_augmentator_loss, ssim_loss, hue_L2, affine_augmentator_loss, affine_L2, rotation_loss, translation_loss, scaling_loss = train_epoch(
            dataloader=train_dataloader,
            primary_model=primary_model,
            hue_augmentator=hue_augmentator,
            affine_augmentator=affine_augmentator,
            # optimizer=optimizer,
            # scheduler=scheduler,
            device=device,
            ssim_weight=ssim_weight,
            rotation_weight=rotation_weight,
            translation_weight=translation_weight,
            scaling_weight=scaling_weight,
            hue_l2_weight=hue_l2_weight,
            affine_l2_weight=affine_l2_weight,
            wandb=wandb,
            epoch=epoch,
            to_be_augmented=to_be_augmented,
            to_save_augmented=to_save_augmented,
            loss_fn=loss_fn,
            overfitting_score=overfitting_score
        )

        val_dict, val_accuracy = val_epoch(
            dataloader=val_dataloader,
            primary_model=primary_model,
            device=device
        )

        # Wandb logging
        wandb_dict = train_dict.copy()
        wandb_dict.update(val_dict)
        overfitting_score = train_accuracy / val_accuracy
        if wandb.run is not None:
            wandb.log(wandb_dict)
        
        primary_losses.append(primary_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        hue_augmentator_losses.append(hue_augmentator_loss)
        ssim_losses.append(ssim_loss)
        hue_L2_losses.append(hue_L2)
        affine_augmentator_losses.append(affine_augmentator_loss)
        affine_L2_losses.append(affine_L2)
        rotation_losses.append(rotation_loss)
        translation_losses.append(translation_loss)
        scaling_losses.append(scaling_loss)

    # Saving the model
    save_model(primary_model)
    if to_be_augmented:
        save_model(hue_augmentator, is_augment=True)
        save_model(affine_augmentator, is_augment=True)
    return (
        min(primary_losses),
        max(train_accuracies),
        min(hue_augmentator_losses),
        min(ssim_losses),
        min(hue_L2_losses),
        min(affine_augmentator_losses),
        min(affine_L2_losses),
        min(rotation_losses),
        min(translation_losses),
        min(scaling_losses),
        max(val_accuracies)
    )