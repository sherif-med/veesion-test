import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.models.lstm_classifier import LSTMClassifier

class PoseClassificationTask(pl.LightningModule):
    def __init__(self, learning_rate=0.001, **model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model : torch.nn.Module = LSTMClassifier(**model_config)
        self.learning_rate = learning_rate


    def training_step(self, batch, batch_idx):
        sequences, labels = batch["seq"], batch["label"]
        logits = self.model(sequences)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch["seq"], batch["label"]
        logits = self.model(sequences)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = batch["seq"], batch["label"]
        logits = self.model(sequences)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        sequences = batch["seq"]
        logits = self.model(sequences)
        preds = torch.argmax(logits, dim=1)
        output = {"class": preds}
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
