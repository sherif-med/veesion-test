import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.models.lstm_video_classifier import LSTMVideoClassifier

class VideoClassificationTask(pl.LightningModule):
    def __init__(self, learning_rate=0.001, **model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTMVideoClassifier(**model_config)
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        video = batch["video"]
        logits = self.model(video)
        loss = F.cross_entropy(logits, batch["label"])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video = batch["video"]
        logits = self.model(video)
        loss = F.cross_entropy(logits, batch["label"])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def predict_step(self, batch, batch_idx):
        image = batch["video"]
        video_class = self.model(image)
        return video_class

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
