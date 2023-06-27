"""
"""

from lion_pytorch import Lion
from pytorch_lightning import LightningModule
from torch.nn import GELU, CrossEntropyLoss, Flatten, Linear, Sequential
from torchmetrics import Accuracy

from ..layers.linear import InverseBottleneck
from ..utils import mixup, mixup_loss


class FlatNet(LightningModule):
    def __init__(
        self,
        image_size: int,
        channels: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        depth: int = 6,
        mixup: bool = False,
        lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mixup = mixup
        self.lr = lr
        self.layers = Sequential(
            Flatten(1, -1),
            Linear(image_size * image_size * channels, hidden_dim),
        )
        for _ in range(depth):
            self.layers.append(
                InverseBottleneck(
                    hidden_dim,
                    expansion_factor=4,
                    dropout=dropout,
                    act=GELU,
                    depth=2,
                )
            )
        self.layers.append(Linear(hidden_dim, num_classes))
        self.crit = CrossEntropyLoss(label_smoothing=0.3)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if mixup:
            x, ya, yb, lam = mixup(x, y, alpha=0.8)
            yh = self(x)
            loss = mixup_loss(self.crit, yh, ya, yb, lam)
        else:
            yh = self(x)
            loss = self.crit(yh, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yh = self(x)
        loss = self.crit(yh, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_accuracy.update(yh, y)
        self.log("test_accu", self.test_accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Lion(self.parameters(), lr=self.lr, weight_decay=0.001)
