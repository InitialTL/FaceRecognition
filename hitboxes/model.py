from .settings import *
from .helper_functions import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class FaceHitbox(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.head = nn.Conv2d(256, 5, kernel_size=1)
        
    def forward(self, X):
        features = self.backbone(X)
        out = self.head(features)
        out = out.permute(0, 2, 3, 1)

        objectness = out[...,  0:1]
        box_coords = torch.sigmoid(out[..., 1:5])
        return objectness, box_coords
