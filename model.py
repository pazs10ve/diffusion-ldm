import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class UNet(nn.Module):
    """
    A simplified variant of the Unet architecture for latent diffusion.
    Operates on latent representations from VAE (7x7 feature maps).
    """
    def __init__(self, latent_dim=56):
        super().__init__()
        image_channels = 32  # Latent space has 32 channels after VAE encoding
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 32  # Output should match latent space dimensions
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection from latent vectors to feature maps
        self.latent_to_features = nn.Linear(latent_dim, 32 * 7 * 7)
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)
        
        # Final projection back to latent vectors
        self.features_to_latent = nn.Linear(32 * 7 * 7, latent_dim)

    def forward(self, x, timestep):
        # If x is latent vectors, convert to feature maps
        if len(x.shape) == 2:  # (batch_size, latent_dim)
            x = self.latent_to_features(x)
            x = x.view(x.size(0), 32, 7, 7)
        
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        
        output = self.output(x)
        
        # Convert back to latent vectors if needed
        output = output.view(output.size(0), -1)
        output = self.features_to_latent(output)
        
        return output


class LatentDiffusionModel(nn.Module):
    def __init__(self, vae, unet, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps
        
        # Set VAE to eval mode and freeze parameters
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Define beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """Calculate the loss for training"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.unet(x_noisy, t)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """Single denoising step"""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Use model to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """Generate samples by iteratively denoising"""
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, batch_size=1, device='cpu'):
        """Generate samples and decode them using VAE"""
        latent_shape = (batch_size, 56)  # latent_dim = 56
        
        # Generate latent samples
        latent_samples = self.p_sample_loop(latent_shape, device)[-1].to(device)
        
        # Decode using VAE
        with torch.no_grad():
            generated_images = self.vae.decode(latent_samples)
        
        return generated_images

    def forward(self, x):
        """Training forward pass"""
        device = x.device
        batch_size = x.shape[0]
        
        # Encode to latent space using VAE
        with torch.no_grad():
            latent = self.vae.encode(x)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # Calculate loss
        loss = self.p_losses(latent, t)
        return loss


