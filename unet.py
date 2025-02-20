import torch
import torch.nn as nn
from embeddings import obj_SinusoidalPostionEmbeddings
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_dims=[32, 64, 128, 256]):
        super().__init__()

        # Initial convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dims[0]),
            nn.GELU()
        )

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 3, padding=1),
                nn.GroupNorm(8, hidden_dims[i+1]),
                nn.GELU(),
                nn.MaxPool2d(2)
            ))

        # Middle
        mid_dim = hidden_dims[-1]
        self.mid = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU()
        )

        # Decoder
        self.ups = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(hidden_dims)-1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i+1], 2, stride=2),
                nn.Conv2d(reversed_dims[i+1], reversed_dims[i+1], 3, padding=1),
                nn.GroupNorm(8, reversed_dims[i+1]),
                nn.GELU()
            ))

        # Output
        self.outc = nn.Conv2d(hidden_dims[0], out_channels, 1)

        # Time embedding
        time_dim = hidden_dims[0]
        self.time_mlp = nn.Sequential(
            obj_SinusoidalPostionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        t = t[..., None, None].expand(-1, -1, x.shape[-2], x.shape[-1])

        # Initial conv
        x = self.inc(x)
        x = x + t

        # Encoder
        skip_connections = []
        for down in self.downs:
            skip_connections.append(x)
            x = down(x)

        # Middle
        x = self.mid(x)

        # Decoder
        for up in self.ups:
            skip = skip_connections.pop()
            x = up(x)
            # Pad if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:])
            x = x + skip

        return self.outc(x)
