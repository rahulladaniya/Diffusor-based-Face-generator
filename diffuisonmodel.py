import torch
import pytorch_lightning as pl
from simpleunet import obje_simpleunet
import torch.nn.functional as F

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

        self.model = obje_simpleunet

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

def obj_diffusionmodel(img_size=64, timesteps=1000):
    return DiffusionModel(img_size, timesteps)