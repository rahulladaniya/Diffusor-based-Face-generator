import torch
import torch.nn as nn
from embeddings import obj_SinusoidalPostionEmbeddings


class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, hidden_dims=[64, 128, 256, 512], time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            obj_SinusoidalPostionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.init_conv = nn.Conv2d(image_channels, hidden_dims[0], 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([])
        for i in range(len(hidden_dims)-1):
            self.downs.append(nn.ModuleList([
                nn.Conv2d(hidden_dims[i], hidden_dims[i], 3, padding=1),
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 3, padding=1, stride=2),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU()
            ]))

        # Middle
        mid_channels = hidden_dims[-1]
        self.mid_block1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.mid_block2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)

        # Upsampling
        self.ups = nn.ModuleList([])
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(hidden_dims)-1):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(reversed_hidden_dims[i],
                                 reversed_hidden_dims[i+1],
                                 4, 2, 1),
                nn.Conv2d(reversed_hidden_dims[i+1] * 2,
                         reversed_hidden_dims[i+1],
                         3, padding=1),
                nn.BatchNorm2d(reversed_hidden_dims[i+1]),
                nn.ReLU()
            ]))

        # Final convolution
        self.final_conv = nn.Conv2d(hidden_dims[0], image_channels, 1)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial convolution
        x = self.init_conv(x)

        # Store skip connections
        skip_connections = []

        # Downsampling
        for down_block in self.downs:
            skip_connections.append(x)
            x = down_block[0](x)
            x = down_block[1](x)
            x = down_block[2](x)
            x = down_block[3](x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        # Upsampling
        for up_block in self.ups:
            skip = skip_connections.pop()
            x = up_block[0](x)  # Upsample
            x = torch.cat((x, skip), dim=1)  # Skip connection
            x = up_block[1](x)  # Conv
            x = up_block[2](x)  # Batch norm
            x = up_block[3](x)  # ReLU

        return self.final_conv(x)


obje_simpleunet=SimpleUNet()