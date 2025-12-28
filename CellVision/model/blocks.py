import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import log_shape

"""
Three backbones (PyTorch) for small grayscale images (25-45 px).
- BetaVAE: encoder -> latent (mu, logvar) -> decoder
- ContrastiveModel: encoder (GAP) -> projection head (for SimCLR/BYOL/SimSiam)
- Hybrid: contrastive encoder -> bottleneck VAE on top (decode from z)

Design goals / notes:
- Use Global Average Pooling (GAP) to reduce positional encoding
- Small capacity to match tiny images (~40x40)
- Inputs: 1-channel grayscale; adjust `in_ch` if needed
- Recommended use: resize/pad images to a stable size (e.g., 40x40 or pad to 56x56 and random crop)

"""

# ---------------------------
# Utility blocks
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, use_bn=False):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)


# ---------------------------
# Encoder (shared ideas)
# ---------------------------
class SmallEncoderGAP(nn.Module):
    """Small convolutional encoder that ends with Global Average Pooling.
    Output is a vector (features) suitable for projection head or VAE bottleneck.
    """
    def __init__(self, in_ch=1, base_filters=16, out_feat=32, input_size=64, trace=False):
        super().__init__()
        # design: conv -> conv(strided) -> conv(strided) -> GAP -> fc
        self.trace = trace

        self.conv1 = ConvBlock(in_ch, base_filters, stride=1) # 64 -> 32
        self.conv2 = ConvBlock(base_filters, base_filters * 2, stride=2)  # downsample 64 -> 32
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4, stride=2)  # downsample 32 -> 16
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8, stride=2)  # downsample 16 -> 8
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 12, stride=2)  # downsample 8 -> 4

        # Compute flatten dim dynamically
        with torch.no_grad():
            x = torch.zeros(1, in_ch, input_size, input_size)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            self.flat_dim = x.numel()

        self.flatten = nn.Flatten()
        # self.gap = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, out_feat),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, C, H, W) expected small H,W
        log_shape("Input", x, self.trace)
        x = self.conv1(x); log_shape("conv1", x, self.trace)
        x = self.conv2(x); log_shape("conv2", x, self.trace)
        x = self.conv3(x); log_shape("conv3", x, self.trace)
        x = self.conv4(x); log_shape("conv4", x, self.trace)
        x = self.conv5(x); log_shape("conv5", x, self.trace)

        x = self.flatten(x); log_shape("flatten", x, self.trace)

        x = self.fc(x); log_shape("fc_out", x, self.trace)
        return x


class SmallDecoder(nn.Module):
    """Simplified decoder with reduced capacity and dropout to force reliance on latent code."""
    def __init__(self, z_dim=16, out_ch=1, base_filters=16, out_size=64, trace=False):
        super().__init__()
        self.trace = trace

        self.out_size = out_size
        
        hidden_ch = base_filters * 4
        
        # Start from 8x8 for more stable upsampling
        self.fc = nn.Linear(z_dim, hidden_ch * 8 * 8)
        
        # Simpler upsampling path with fewer blocks and dropouts
        self.up = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(hidden_ch, base_filters * 2),
            # nn.Dropout2d(0.15),  # Dropout to prevent memorization
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_filters * 2, base_filters ),
            # nn.Dropout2d(0.1),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_filters , base_filters),
            # nn.Dropout2d(0.1),

            # Extra convblock for some detail?
            ConvBlock(base_filters, base_filters),
            
            # Final conv to output channels
            nn.Conv2d(base_filters, out_ch, kernel_size=3, stride=1, padding=1),

            nn.Sigmoid()
        )

    def forward(self, z):
        # z: (B, z_dim)
        log_shape("z", z, self.trace)

        x = self.fc(z); log_shape("fc", x, self.trace)

        B = z.size(0)
        x = x.view(B, -1, 8, 8)  # Reshape to (B, hidden_ch, 4, 4)
        
        log_shape("reshape_to_4x4", x, self.trace)
        for i, layer in enumerate(self.up):
            x = layer(x)
            log_shape(f"up[{i}] ({layer.__class__.__name__})", x, self.trace)
        return x
    

# ---------------------------
# Contrastive backbone + projection head
# ---------------------------
class ProjectionHead(nn.Module):
    """Simple MLP projection head used in SimCLR/BYOL etc.
    maps encoder features to projection space for contrastive loss.
    """
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)