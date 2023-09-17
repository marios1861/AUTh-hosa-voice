import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Accuracy


class VoiceClassifier(pl.LightningModule):
    def __init__(self, backbone, lr):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.tot_accuracy = Accuracy(task="multiclass", num_classes=4)
        self.accuracy = Accuracy(task="multiclass", num_classes=4, average="none")

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.backbone(x)
        loss = self.loss(preds, y)
        self.tot_accuracy(preds, y)
        self.accuracy(preds, y)
        accuracies = self.accuracy
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc",
            self.tot_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="tot_accuracy",
        )
        self.log(
            "train/acc_0",
            accuracies[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "train/acc_1",
            accuracies[1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "train/acc_2",
            accuracies[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "train/acc_3",
            accuracies[3],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="accuracy",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.backbone(x)
        loss = self.loss(preds, y)
        self.tot_accuracy(preds, y)
        self.accuracy(preds, y)
        accuracies = self.accuracy
        self.log("val/loss", loss, prog_bar=True)
        self.log(
            "val/acc",
            self.tot_accuracy,
            prog_bar=True,
            metric_attribute="tot_accuracy",
        )
        self.log(
            "val/acc_0",
            accuracies[0],
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "val/acc_1",
            accuracies[1],
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "val/acc_2",
            accuracies[2],
            prog_bar=True,
            metric_attribute="accuracy",
        )
        self.log(
            "val/acc_3",
            accuracies[3],
            prog_bar=True,
            metric_attribute="accuracy",
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.backbone(x)
        loss = self.loss(preds, y)
        self.tot_accuracy(preds, y)
        self.accuracy(preds, y)
        accuracies = self.accuracy
        self.log("test/loss", loss, prog_bar=True)
        self.log(
            "test/acc",
            self.tot_accuracy,
            metric_attribute="tot_accuracy",
        )
        self.log(
            "test/acc_0",
            accuracies[0],
            metric_attribute="accuracy",
        )
        self.log(
            "test/acc_1",
            accuracies[1],
            metric_attribute="accuracy",
        )
        self.log(
            "test/acc_2",
            accuracies[2],
            metric_attribute="accuracy",
        )
        self.log(
            "test/acc_3",
            accuracies[3],
            metric_attribute="accuracy",
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
