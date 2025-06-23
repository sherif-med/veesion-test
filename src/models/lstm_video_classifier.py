from torch import nn
import torch
from src.tasks.ssl_frame import SSLFrameTask
from src.models.frame_encoder import FrameEncoder

class LSTMVideoClassifier(nn.Module):

    def __init__(self, frame_encoder_ckpt=None, hidden_dim=256, num_classes=10, num_layers=2, projection_dim=None):
        super().__init__()
        if frame_encoder_ckpt is not None:
            ssl_task = SSLFrameTask.load_from_checkpoint(frame_encoder_ckpt)
            self.frame_encoder_model = ssl_task.model
        else:
            assert projection_dim is not None, "If frame_encoder_ckpt is not provided, projection_dim must be provided"
            self.frame_encoder_model : torch.nn.Module = FrameEncoder(projection_dim)

        self.lstm = nn.LSTM(self.frame_encoder_model.projection_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, video: torch.Tensor):
        # video shape is (N, T, C, H, W)
        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)          # Shape: (B*T, C, H, W)
        features = self.frame_encoder_model(x)  # Output: (B*T, F)
        features = features.view(B, T, -1)      # Reshape back: (B, T, F)

        # Apply LSTM
        # Initialize hidden and cell states
        # h0 shape: (num_layers, batch_size, hidden_size)
        # c0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.lstm.num_layers, video.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, video.size(0), self.lstm.hidden_size).to(x.device)

        # Pass through LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        # (hn, cn) are the hidden and cell states for the last time step
        out, (hn, cn) = self.lstm(features, (h0, c0))

        out = hn[-1, :, :]

        logits = self.fc(out)
        return logits