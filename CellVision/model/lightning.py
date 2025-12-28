import lightning as L
import torch
import torch.nn.functional as F

class DenoisingBetaVAE_Lit(L.LightningModule):
    def __init__(self, vae, beta=4.0, lr=1e-3):
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.lr = lr
        self.warmup_epochs=5

    def training_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(target)

        # # --- recon loss ---
        # if self.recon_type == "mse":
        #     recon_loss = F.mse_loss(recon, target, reduction="sum")
        # else:
        recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / len(batch)

        # --- KL ---
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl.mean()

        # --- total loss ---
        loss = recon_loss + self.beta * kl

        self.log("train_reconstruction", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_KL", kl, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_totalloss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(target)

        recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / len(batch)

        # --- KL ---
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl.mean()

        loss = recon_loss + self.beta * kl
        self.log("val_reconstruction", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_KL", kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_totalloss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['aug_1']
        target = batch['targets']

        recon, z, mu, logvar = self.vae(target)

        recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / len(batch)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl

        self.log("test_reconstruction", recon_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_KL", kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_totalloss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Optional: Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Optional: Log gradient norms at end of epoch."""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log('train/grad_norm', total_norm)