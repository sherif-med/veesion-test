from torch import nn
import torchvision

class FrameEncoder(nn.Module):

    def __init__(self, projection_dim):
        super().__init__()
        encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])  # Remove FC
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)#.squeeze()
        z = self.projector(h)
        return z