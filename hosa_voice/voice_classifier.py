import lightning.pytorch as pl
import torch
from torch import nn


class VoiceClassifier(pl.LightningModule):
    def __init__(self, backbone, lr, freeze_stage):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.lr = lr
        self.freeze_stage = freeze_stage
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.backbone(x), y)
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
