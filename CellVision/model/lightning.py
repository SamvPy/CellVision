import lightning as L
import torch
import torch.nn.functional as F

class DenoisingBetaVAE_Lit(L.LightningModule):
    def __init__(self, vae, beta=4.0, lr=1e-3, recon_type="mse"):
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.lr = lr
        self.recon_type = recon_type

    def training_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(x)

        # --- recon loss ---
        if self.recon_type == "mse":
            recon_loss = F.mse_loss(recon, target, reduction="sum")
        else:
            recon_loss = F.binary_cross_entropy_with_logits(recon, target, reduction="sum")

        # --- KL ---
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # --- total loss ---
        loss = recon_loss + self.beta * kl

        self.log("train_reconstruction", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_KL", kl, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_totalloss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(x)

        recon_loss = F.mse_loss(recon, target, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl

        self.log("val_reconstruction", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_KL", kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_totalloss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(x)

        recon_loss = F.mse_loss(recon, target, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl

        self.log("test_reconstruction", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_KL", kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_totalloss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
