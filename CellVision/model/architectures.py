import torch
import torch.nn as nn
from .blocks import SmallEncoderGAP, SmallDecoder

class BetaVAE(nn.Module):
    """Small beta-VAE using SmallEncoderGAP as encoder trunk.
    Encoder returns mu, logvar. Decoder reconstructs canonical centered image.
    """
    def __init__(self, in_ch=1, z_dim=8, base_filters=16, hidden_feat=16, out_size=64, beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder_trunk = SmallEncoderGAP(in_ch=in_ch, base_filters=base_filters, out_feat=hidden_feat)
        self.fc_mu = nn.Linear(hidden_feat, z_dim)
        self.fc_logvar = nn.Linear(hidden_feat, z_dim)
        self.decoder = SmallDecoder(z_dim=z_dim, out_ch=in_ch, base_filters=base_filters, out_size=out_size)

    def encode(self, x):
        h = self.encoder_trunk(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar