import torch
import torch.nn as nn
from src.models.frame_encoder import FrameEncoder
from src.tasks.ssl_frame import SSLFrameTask

class TransformerVideoClassifier(nn.Module):
    def __init__(self, frame_encoder_ckpt=None, nhead=8, dim_feedforward=2048, dropout=0.1, num_classes=10, projection_dim=None):
        super().__init__()
        if frame_encoder_ckpt is not None:
            ssl_task = SSLFrameTask.load_from_checkpoint(frame_encoder_ckpt)
            self.frame_encoder_model = ssl_task.model
        else:
            assert projection_dim is not None, "If frame_encoder_ckpt is not provided, projection_dim must be provided"
            self.frame_encoder_model: torch.nn.Module = FrameEncoder(projection_dim)

        self.temporal_transformer = nn.TransformerEncoderLayer(
            d_model=self.frame_encoder_model.projection_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
            )

        self.fc = nn.Linear(self.frame_encoder_model.projection_dim, num_classes)

    def forward(self, video: torch.Tensor):
        # video shape is (N, T, C, H, W)
        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)  # Shape: (B*T, C, H, W)
        features = self.frame_encoder_model(x)  # Output: (B*T, F)
        features = features.view(B, T, -1)  # Reshape back: (B, T, F)

        # Apply Temporal Transformer
        x = self.temporal_transformer(features)
        x = x.mean(dim=1)  # (B, T, F) -> (B, F)
        x = self.fc(x)

        return x