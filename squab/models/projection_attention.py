"""
"""

from typing import List, Literal

import torch
from pytorch_lightning import LightningModule
from torch.nn import Dropout, Linear, LogSoftmax, ModuleList, Tanh
from torch.nn.functional import nll_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torchmetrics import Accuracy

from ..layers.projection import ProjAttention


class ProjectionAttention(LightningModule):
    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        out_channels: int,
        out_size: int,
        masks: List[str],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embed = Linear(feature_dim, embedding_dim)
        self.drop = Dropout(dropout)
        self.proj = ModuleList(
            [ProjAttention(1, out_channels, embedding_dim, mask=mask) for mask in masks]
        )
        self.out = Linear(out_channels * len(masks), out_size)
        self.tanh = Tanh()
        self.ls = LogSoftmax()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=out_size)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=out_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=out_size)

    def forward(self, x):
        xh = self.drop(self.tanh(self.embed(x)))
        xh = xh.unsqueeze(1)
        fs = self.tanh(torch.cat([layer(xh) for layer in self.proj], dim=1))
        return self.ls(self.out(fs))

    def training_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = nll_loss(logits, y.flatten())
        self.train_accuracy.update(logits, y.flatten())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.log("train_accuracy_epoch", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = nll_loss(logits, y.flatten())
        self.val_accuracy.update(logits, y.flatten())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_val_batch_end(self, *args, **kwargs):
        self.log("val_accuracy_epoch", self.val_accuracy.compute(), prog_bar=True)
        self.val_accuracy.reset()

    def test_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)

        loss = nll_loss(logits, y.flatten())
        self.test_accuracy.update(logits, y.flatten())
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_batch_end(self, *args, **kwargs):
        self.log("test_accuracy_epoch", self.test_accuracy.compute())
        self.test_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=0.01,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
