import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.frame_encoder import FrameEncoder

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.mm(representations, representations.t())
    labels = torch.arange(z1.size(0)).repeat(2).to(z1.device)

    logits = similarity / temperature
    logits = logits - torch.eye(logits.size(0), device=logits.device) * 1e9  # mask self-similarity
    return nn.CrossEntropyLoss()(logits, labels)


class SSLFrameTask(pl.LightningModule):
    def __init__(self, learning_rate=0.001, **model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model : torch.nn.Module = FrameEncoder(**model_config)
        self.learning_rate = learning_rate


    def training_step(self, batch, batch_idx):
        pair_left = batch["pair_left"]
        pair_right = batch["pair_right"]
        z1 = self.model(pair_left)
        z2 = self.model(pair_right)
        loss = nt_xent_loss(z1, z2)
        self.log('ssl_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        embedding = self.model(image)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
