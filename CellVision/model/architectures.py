import torch
import torch.nn as nn
from .blocks import SmallEncoderGAP, SmallDecoder
from .utils import log_shape

class BetaVAE(nn.Module):
    """Small beta-VAE using SmallEncoderGAP as encoder trunk.
    Encoder returns mu, logvar. Decoder reconstructs canonical centered image.
    """
    def __init__(self, in_ch=1, z_dim=8, base_filters=16, hidden_feat=16, out_size=64, beta=1.0, trace=False):
        super().__init__()
        self.trace = trace
        
        self.beta = beta
        self.encoder_trunk = SmallEncoderGAP(in_ch=in_ch, base_filters=base_filters, out_feat=hidden_feat, trace=trace)
        self.fc_mu = nn.Linear(hidden_feat, z_dim)
        self.fc_logvar = nn.Linear(hidden_feat, z_dim)
        self.decoder = SmallDecoder(z_dim=z_dim, out_ch=in_ch, base_filters=base_filters, out_size=out_size, trace=trace)

    def encode(self, x):
        h = self.encoder_trunk(x)
        log_shape("encoder_out", h, self.trace)
        mu = self.fc_mu(h); log_shape("mu", mu, self.trace)
        logvar = self.fc_logvar(h); log_shape("logvar", logvar, self.trace)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        log_shape("z", z, self.trace)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        log_shape("recon", recon, self.trace)
        return recon, z, mu, logvar