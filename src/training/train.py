"""
src/training/train.py
Main training entry point.

Usage:
    python src/training/train.py --config configs/resnet18.yaml
    python src/training/train.py --config configs/efficientnet_b2.yaml
    python src/training/train.py --config configs/vit_dino.yaml
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import numpy as np

from src.models.model_factory import build_model
from src.data.dataset import FASDataModule
from src.evaluation.metrics import compute_all_metrics


# ─────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────

class FASLightningModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterion = nn.CrossEntropyLoss()

        self.val_preds  = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]   # prob of being real

        self.val_preds.append(probs.cpu().numpy())
        self.val_labels.append(labels.cpu().numpy())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        all_preds  = np.concatenate(self.val_preds)
        all_labels = np.concatenate(self.val_labels)

        metrics = compute_all_metrics(all_labels, all_preds)

        self.log("val_acer", metrics["acer"])
        self.log("val_auc",  metrics["auc"])
        self.log("val_eer",  metrics["eer"])
        self.log("val_hter", metrics["hter"])

        print(f"\n  Val → ACER: {metrics['acer']}% | AUC: {metrics['auc']}% | "
              f"EER: {metrics['eer']}% | HTER: {metrics['hter']}%")

        self.val_preds  = []
        self.val_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs,
            eta_min=1e-6,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_train_epoch_start(self):
        # Unfreeze backbone after warmup epochs (for fine-tuning models)
        freeze_epochs = self.cfg.model.get("freeze_backbone_epochs", 0)
        if freeze_epochs > 0 and self.current_epoch == freeze_epochs:
            self.model.unfreeze_backbone()


# ─────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────

def train(config_path: str):
    cfg = OmegaConf.load(config_path)

    pl.seed_everything(cfg.project.seed)

    # Logger
    logger = WandbLogger(
        project=cfg.project.name,
        entity=cfg.project.wandb_entity,
        name=cfg.model.name,
        config=OmegaConf.to_container(cfg),
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.paths.checkpoints,
        filename=f"{cfg.model.name}" + "-{epoch:02d}-{val_acer:.2f}",
        monitor=cfg.logging.monitor,
        mode=cfg.logging.mode,
        save_top_k=cfg.logging.save_top_k,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg.logging.monitor,
        patience=cfg.training.early_stopping_patience,
        mode=cfg.logging.mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Data
    dm = FASDataModule(cfg)

    # Model
    module = FASLightningModule(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.get("precision", 32),
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
    )

    trainer.fit(module, dm.train_dataloader(), dm.val_dataloader())

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val ACER:   {checkpoint_cb.best_model_score:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
