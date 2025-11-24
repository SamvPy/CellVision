import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, use_bn=True):
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
    def __init__(self, in_ch=1, base_filters=16, out_feat=32):
        super().__init__()
        # design: conv -> conv(strided) -> conv(strided) -> GAP -> fc
        self.conv1 = ConvBlock(in_ch, base_filters, stride=1)
        self.conv2 = ConvBlock(base_filters, base_filters * 2, stride=2)  # downsample
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4, stride=2)  # downsample
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8, stride=2)  # downsample
        self.gap = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(base_filters * 8, out_feat),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, C, H, W) expected small H,W
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)  # (B, channels)
        x = self.fc(x)
        return x


# ---------------------------
# Decoder for VAE (simple upsampling)
# ---------------------------
class SmallDecoder(nn.Module):
    """Simple decoder that maps latent z to image of specified size.
    Uses linear -> reshape -> convtranspose / upsample conv.
    """
    def __init__(self, z_dim=16, out_ch=1, base_filters=16, out_size=64):
        super().__init__()
        # compute a small spatial size to reshape into
        # we'll reshape into (base_filters*4, h', w') where h'*w' approx = out_size//4 square
        self.out_size = out_size
        # choose a fixed small spatial grid: e.g., 5x5 if out_size ~40 (5*8=40 with upsample)
        # We'll use a simple decoder that upsamples twice.
        hidden_ch = base_filters * 8
        self.fc = nn.Linear(z_dim, hidden_ch * 8 * 8)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(hidden_ch, base_filters * 4),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_filters * 4, base_filters * 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(base_filters * 2, base_filters),
            nn.Conv2d(base_filters, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, -1, 8, 8)  # (B, hidden_ch, 5, 5)
        x = self.up(x)
        # now x is larger than 40x40 depending on upsample specifics; center-crop / interpolate to out_size
        # x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
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