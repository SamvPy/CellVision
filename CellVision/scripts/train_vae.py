import yaml
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from pathlib import Path
import torch
import lightning.pytorch as pl
import argparse
import multiprocessing as mp
import subprocess
import signal
import os

from CellVision.data.dataset import ImageDataset
from CellVision.data.datamodules import ImageDataModule

from CellVision.model.architectures import BetaVAE
from CellVision.model.lightning import DenoisingBetaVAE_Lit


def train(cfg):
    """One training run given a full config dict."""
    # Speed up
    torch.set_float32_matmul_precision('medium')

    # Set up logger
    wandb_logger = WandbLogger(
        project=cfg["wandb"].get("project", "CellVision-2025"),
        name=cfg["wandb"].get("run_name", "Undefined"),
        config=cfg,
        save_dir=cfg["wandb"].get("save_dir")
    )

    # Reproducibility
    pl.seed_everything(cfg.get("seed", 42))
    
    # === Data ===
    datamodule = ImageDataModule(
        data_dir=cfg['data']['root'],
        batch_size=cfg['data']['batch_size'],
        name_dataset_val=cfg['data']['val'],
        name_dataset_train=cfg['data']['train'],
        name_dataset_test=cfg['data']['test'],
        num_workers=cfg['data']['num_workers'],
        dataset_cls=ImageDataset
    )

    # === Model ===
    model = DenoisingBetaVAE_Lit(
        vae=BetaVAE(
            in_ch=cfg['model']['in_channels'],
            z_dim=cfg['model']['z_dim'],
            base_filters=cfg['model']['base_filters'],
            hidden_feat=cfg['model']['hidden_features'],
            out_size=cfg['model']['out_size'],
            beta=cfg['model']['beta']
        ),
        beta=cfg['model']['beta'],
        lr=cfg['model']['lr']
    )

    # === Trainer ===
    trainer = Trainer(
        logger=wandb_logger,
        **cfg['trainer']
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train Lightning model with config or run sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, choices=["train", "sweep"], default="train")
    parser.add_argument("--sweep", type=str, help="Path to sweep YAML (only in sweep mode)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg)
    # elif args.mode == "sweep":
    #     if not args.sweep:
    #         raise ValueError("Must provide --sweep when mode is 'sweep'")
    #     run_sweep_mode(args.config, args.sweep)
