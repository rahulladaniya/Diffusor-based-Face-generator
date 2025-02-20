import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from PIL import Image
import math

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, hidden_dims=[64, 128, 256, 512], time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
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
            SinusoidalPositionEmbeddings(time_dim),
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

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class DiffusionModel(pl.LightningModule):
    def __init__(self, img_size=64, timesteps=1000):
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps

        # Define beta schedule
        beta = self.cosine_beta_schedule(timesteps)
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # Register buffers to automatically move them to the correct device
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)

        self.model = SimpleUNet()

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
        alpha_hat = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_hat = alpha_hat / alpha_hat[0]
        beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        return torch.clip(beta, 0.0001, 0.02)

    def get_loss(self, batch, batch_idx):
        # Ensure batch is on the correct device
        x = batch.to(self.device)
        # Generate random timesteps on the correct device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device)
        noise = torch.randn_like(x, device=self.device)

        x_t = (
            torch.sqrt(self.alpha_hat[t])[:, None, None, None] * x +
            torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] * noise
        )

        predicted_noise = self.model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    @torch.no_grad()
    def sample(self, n_samples, guidance_scale=3.0):
        self.eval()
        # Create initial noise on the correct device
        x = torch.randn(n_samples, 3, self.img_size, self.img_size, device=self.device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i], device=self.device).repeat(n_samples)

            predicted_noise = self.model(x, t)

            if i > 0:
                noise = torch.randn_like(x, device=self.device)
            else:
                noise = torch.zeros_like(x, device=self.device)

            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        x = torch.clamp(x, -1, 1)
        self.train()
        return x



class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size=64, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        self.dataset = FaceDataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )


def train_model(data_dir, img_size=64, batch_size=8, max_epochs=100, checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    data_module = FaceDataModule(
        data_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    model = DiffusionModel(img_size=img_size)

    # Define multiple checkpoint callbacks
    checkpoints = [
        # Save best models based on training loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-checkpoint-{epoch:02d}-{train_loss:.2f}',
            monitor='train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        # Save periodic checkpoints every 5 epochs
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='periodic-checkpoint-{epoch:02d}',
            every_n_epochs=5,
            save_on_train_epoch_end=True,
        ),
        # Save last checkpoint
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='last-checkpoint',
            save_last=True,
        )
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=checkpoints,
        gradient_clip_val=1.0,
        precision="16-mixed",
        accumulate_grad_batches=4,
        enable_progress_bar=True,
        logger=True
    )

    # Train the model
    trainer.fit(model, data_module)
    
    # Save the final model state
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'img_size': img_size,
        'timesteps': model.timesteps
    }, final_model_path)
    
    return model

# Function to load the saved model


if __name__ == "__main__":
    # Set parameters
    DATA_DIR = "img_align_celeba"  # Update this path to your dataset location
    IMG_SIZE = 64
    BATCH_SIZE = 8
    MAX_EPOCHS = 30

    # Verify data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist!")

    # Print number of images found
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {DATA_DIR}")

    # Train model
    model = train_model(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )

    