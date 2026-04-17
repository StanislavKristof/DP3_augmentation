"""This file reads data from config, creates a model and trains it."""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from model import get_models
from data_module import get_data
import torch
from trainer import train_models
import wandb
from torchinfo import summary
from termcolor import colored
import pystache
from ruamel.yaml import YAML
from icecream import ic
from utils import (
    add_colab_path,
    get_device,
    is_in_colab,
    log_code_from,
    prepare_config_hydra_yaml,
    update_version
)
import optuna
from configparser import ConfigParser

def do_training(trial, cfg):
    epochs = cfg.config.train_info.epochs
    num_workers = cfg.config.train_info.num_workers
    project = cfg.config.train_info.project
    name = cfg.config.train_info.name
    version = cfg.config.train_info.version
    device = cfg.config.train_info.device
    wandb_on = cfg.config.train_info.wandb
    image_size = cfg.datasets.image_size
    to_be_augmented = cfg.config.train_info.to_be_augmented
    to_save_augmented = cfg.config.train_info.to_save_augmented
    batch_size = cfg.config.train_info.batch_size
    hue_augmentator_learning_rate = cfg.config.train_info.hue_augmentator_learning_rate
    affine_augmentator_learning_rate = cfg.config.train_info.hue_augmentator_learning_rate
    ssim_weight = cfg.config.train_info.ssim_weight
    rotation_weight = cfg.config.train_info.rotation_weight
    translation_weight = cfg.config.train_info.translation_weight
    scaling_weight = cfg.config.train_info.scaling_weight
    hue_l2_weight = cfg.config.train_info.hue_l2_weight
    affine_l2_weight = cfg.config.train_info.affine_l2_weight
    scaling_bound = cfg.config.train_info.scaling_bound
    translation_bound = cfg.config.train_info.translation_bound
    rotation_bound = cfg.config.train_info.rotation_bound
    primary_learning_rate = cfg.config.train_info.primary_learning_rate
    padding_mode = cfg.config.train_info.padding_mode
    primary_weight_decay = cfg.config.train_info.primary_weight_decay

    hue_augmentator_learning_rate = trial.suggest_float('hue_augmentator_learning_rate', hue_augmentator_learning_rate.min_value, hue_augmentator_learning_rate.max_value)
    affine_augmentator_learning_rate = trial.suggest_float('affine_augmentator_learning_rate', affine_augmentator_learning_rate.min_value, affine_augmentator_learning_rate.max_value)
    ssim_weight = trial.suggest_float('ssim_weight', ssim_weight.min_value, ssim_weight.max_value)
    hue_l2_weight = trial.suggest_float('hue_l2_weight', hue_l2_weight.min_value, hue_l2_weight.max_value)
    affine_l2_weight = trial.suggest_float('affine_l2_weight', affine_l2_weight.min_value, affine_l2_weight.max_value)
    rotation_weight = trial.suggest_float('rotation_weight', rotation_weight.min_value, rotation_weight.max_value)
    translation_weight = trial.suggest_float('translation_weight', translation_weight.min_value, translation_weight.max_value)
    scaling_weight = trial.suggest_float('scaling_weight', scaling_weight.min_value, scaling_weight.max_value)
    scaling_bounds = {"min": scaling_bound.min_value, "max": scaling_bound.max_value}
    translation_bounds = {"min": translation_bound.min_value, "max": translation_bound.max_value}
    rotation_bounds = {"min": rotation_bound.min_value, "max": rotation_bound.max_value}
    primary_learning_rate = trial.suggest_float('primary_learning_rate', primary_learning_rate.min_value, primary_learning_rate.max_value)
    # primary_weight_decay = trial.suggest_float('primary_weight_decay', primary_weight_decay.min_value, primary_weight_decay.max_value)
    to_track = dict(cfg["config"])
    to_track.update(trial.params)
    if wandb_on:
        config = ConfigParser()
        config.read("credentials.ini")
        wandb_api_key = config["WandB"]["api_key"]
        wandb.login(key=wandb_api_key)
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            entity="DP_augmentation_team",
            project=project,

            # track hyperparameters and run metadata
            config=to_track,
            name=f"{name}.{trial.number + 1}"
        )
        wandb.run.log_code(log_code_from())

    # Get cpu, gpu or mps device for training.
    device = get_device(device=device)
    print(f"Using {device} device")

    primary_model, hue_augmentator, affine_augmentator = get_models(
        device=device,
        feature_extractor=cfg.config.model_info.feature_extractor,
        hue_augmentator=cfg.config.model_info.hue_augmentator,
        number_of_classes=cfg.datasets.number_of_classes,
        image_size=cfg.datasets.image_size,
        primary_learning_rate=primary_learning_rate,
        primary_weight_decay=primary_weight_decay,
        hue_augmentator_learning_rate=hue_augmentator_learning_rate,
        affine_augmentator_learning_rate=affine_augmentator_learning_rate,
        # augmentator_weight_decay=augmentator_weight_decay,
        epochs=epochs,
        scaling_bounds=scaling_bounds,
        translation_bounds=translation_bounds,
        rotation_bounds=rotation_bounds,
        padding_mode=padding_mode
    )

    summary(
        primary_model,
        input_size=(batch_size, 3, image_size, image_size),
        device=device
    )

    if to_be_augmented:
        summary(
            hue_augmentator,
            input_size=(batch_size, 3, image_size, image_size),
            device=device
        )
        summary(
            affine_augmentator,
            input_size=(batch_size, 3, image_size, image_size),
            device=device
        )

    if wandb.run is None:
        message = "Wandb is OFF! Automatic version update is turned off!"
        print(colored(message, "light_red"))

    train_dataloader, test_dataloader = get_data(
        batch_size, num_workers, cfg.datasets
    )

    primary_loss, train_accuracy, hue_augmentator_loss, ssim_loss, hue_L2_loss, affine_augmentator_loss, affine_L2_loss, rotation_loss, translation_loss, scaling_loss, val_accuracy = train_models(
        epochs=epochs,
        primary_model=primary_model,
        hue_augmentator=hue_augmentator,
        affine_augmentator=affine_augmentator,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        device=device,
        number_of_classes=cfg.datasets.number_of_classes,
        image_size=image_size,
        ssim_weight=ssim_weight,
        rotation_weight=rotation_weight,
        translation_weight=translation_weight,
        scaling_weight=scaling_weight,
        hue_l2_weight=hue_l2_weight,
        affine_l2_weight=affine_l2_weight,
        wandb=wandb,
        to_be_augmented=to_be_augmented,
        to_save_augmented=to_save_augmented
    )

    wandb.finish()
    return val_accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """Get data from config, use them to create a model, train and save it."""
    trials = cfg.config.train_info.trials
    config = dict(cfg["config"])

    study = optuna.create_study()
    study.set_user_attr("cfg", cfg)
    study.optimize(lambda trial: do_training(trial, cfg), n_trials=trials)


if __name__ == "__main__":
    config_yaml = "conf/config.yaml"

    my_app()

    initialize(
        version_base=None,
        config_path="conf",
        job_name="get_wandb_state"
    )

    cfg = compose(config_name="config", return_hydra_config=True)
    is_wandb_on = cfg["config"]["train_info"]["wandb"]

    if is_wandb_on:
        update_version(config_yaml)

    wandb.finish()
