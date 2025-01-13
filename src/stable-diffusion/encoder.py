import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock

class Encoder(nn.Sequential):
    def __init__(self):
        super(Encoder, self).__init__(
            # downsample the image by a factor of 8
            
            # (Batch_Size, 3, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=2),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=2),
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=2),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=2),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(num_groups=32, num_channels=512),

            # sigmoid linear unit - which was used in Kingma & Welling
            nn.SiLU(),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)        
            nn.conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (Batch_Size, Channels, Height, Width)
        # noise (Batch_Size, Channels, Height / 8, Width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height, Width)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (Batch_Size, 8, Height / 8, Width / 8) -> 2 x (Batch_Size, 4, Height / 8, Width / 8)
        # these are the parameters that are representing the latent space of the VAE
        mu, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, min=-30.0, max=20.0)
        var = torch.exp(log_var)

        std = torch.sqrt(var)

        x = mu + std * noise
        x *= 0.18215

        return x



        
