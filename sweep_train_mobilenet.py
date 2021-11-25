"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb

from src.dataloader import create_sweep_dataloader
from src.loss import CustomCriterion, F1Loss, FocalLoss
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
import torch.quantization
import timm
import random
import numpy as np
import torchvision


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    print(f"Set seed to {seed}")


def train(
    config=None,
) -> Tuple[float, float, float]:
    """Train."""
    wandb.init(name="mobilenetv2_int8")
    if config == None:
        config = wandb.config
    # float32_model = timm.create_model(
    #     model_name="tf_efficientnet_lite0", pretrained=True, num_classes=6
    # )
    # model = torch.quantization.quantize_dynamic(
    #     float32_model, {nn.Conv2d, nn.Linear, nn.BatchNorm2d}, dtype=torch.qint8
    # )
    model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    model = model.to(torch.device("cuda:0"))
    print(model.parameters())
    # Create dataloader
    train_dl, val_dl, test_dl = create_sweep_dataloader(config)

    # Create optimizer, scheduler, criterion
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["lr_default"], momentum=0.9
        )
    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_default"])
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config["lr_default"],
        )
    print("new scheduler")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=config["factor"],
        patience=config["patience"],
        verbose=True,
    )
    if config["loss"] == "focal":
        criterion = FocalLoss()
    elif config["loss"] == "weightedfocal":
        criterion = FocalLoss(
            weight=torch.tensor(
                [1.45, 0.35, 1.66, 0.63, 0.34, 1.55], dtype=torch.half
            ).to(torch.device("cuda:0"))
        )
    elif config["loss"] == "f1":
        criterion = F1Loss()
    elif config["loss"] == "ce":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "weightedce":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.45, 0.35, 1.66, 0.63, 0.34, 1.55], dtype=torch.half
            ).to(torch.device("cuda:0"))
        )

    # Amp loss scaler
    scaler = torch.cuda.amp.GradScaler()

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cuda:0"),
        model_path="exp/sweep/best.pt",
        verbose=1,
        sweep=True,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=config["max_epochs"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load("exp/sweep/best.pt"))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    opt = yaml.load(
        open("/opt/ml/code/configs/sweep_config.yaml"), Loader=yaml.FullLoader
    )
    opt.update(vars(args))
    args = opt
    print("arguments: {}".format(str(args)))
    # if os.path.exists(log_dir):
    #     modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
    #     new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
    #     os.rename(log_dir, new_log_dir)
    # sweep_id = wandb.sweep(args, project="lightweight", entity="passion-ate")
    wandb.agent("passion-ate/lightweight/573d212o", train, count=100)
