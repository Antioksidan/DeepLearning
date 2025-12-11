import torch
import torch.nn as nn
from torchvision import models, utils
import torch.nn.functional as F
import numpy as np
import math
import tqdm
import os, time
import wandb

# custom dataloader
from src.dataloader import DIV2KDataModule
# Generator network
from src.esrgan.generator2 import Generator_RRDB
# Discriminator network
from src.discriminator import Discriminator
# module for VGG-based perceptual loss
from src.esrgan.vgg_wrapper2 import VGGLoss2

from src.training_functions import psnr, save_sample

# =========================================================
# TRAINING
# =========================================================
def pretrain_generator_rrdb(G, train_loader, pretrain_epochs=10, lr=1e-4, save_samples=False, use_wandb=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize wandb if enabled
    if use_wandb:
        timestamp = int(time.time())
        wandb.init(
            name=f"pretrain_generator_ESRGAN_{timestamp}",
            project="veo_srgan", 
            config={
            "pretrain_epochs": pretrain_epochs,
            "device": device,
            "lr": lr,
            "architecture": "Generator_RRDB",
        })
        wandb.watch(G, log="all", log_freq=100)

    # pretrain generator (with mae)
    print("Pretraining generator...")

    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    mae = nn.L1Loss()
    mae.to(device)

    for epoch in range(1, pretrain_epochs + 1):
        G.train()
        loop = tqdm.tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{pretrain_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optim_G.zero_grad()
            sr_img = G(lr_img)

            pixel_loss = mae(sr_img, hr_img)
            pixel_loss.backward()
            optim_G.step()

            epoch_loss += pixel_loss.item()
            num_batches += 1

            loop.set_postfix({
                "G": f"{pixel_loss.item():.4f}",
            })

            # Log batch metrics
            if use_wandb:
                wandb.log({"pretrain/batch_loss": pixel_loss.item()})

        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        if use_wandb:
            wandb.log({
                "pretrain/epoch": epoch,
                "pretrain/avg_loss": avg_loss,
            })

        # Save sample image from last batch
        if save_samples:
            save_sample(sr_img, epoch, out_dir="pretrain_samples")
            if use_wandb:
                wandb.log({"pretrain/sample": wandb.Image(sr_img[0].detach().cpu())})
    
    if use_wandb:
        wandb.finish()
    
    return G

def train_esrgan(G, D, train_loader, val_loader=None, 
          num_epochs=50, 
          lr=1e-4, 
          w_pix=1e-2,
          w_adv=5e-3, 
          save_samples=False, 
          use_wandb=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize wandb if enabled
    if use_wandb:
        timestamp = int(time.time())
        wandb.init(
            name=f"train_ESRGAN_{timestamp}",
            project="veo_srgan", 
            config={
            "num_epochs": num_epochs,
            "device": device,
            "lr": lr,
            "w_pix": w_pix,
            "w_adv": w_adv,
            "architecture": "ESRGAN",
        })
        wandb.watch([G, D], log="all", log_freq=100)

    vgg_loss = VGGLoss2().to(device)
    bce = nn.BCEWithLogitsLoss().to(device)
    mae = nn.L1Loss().to(device)

    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    optim_D = torch.optim.Adam(D.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()

        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # ---------------------
            # Train Discriminator (RaGAN)
            # ---------------------
            optim_D.zero_grad()

            # Predict on HR (real) and SR (fake)
            real_out = D(hr_img)
            sr_img = G(lr_img).detach() # detach G gradients for D training
            fake_out = D(sr_img)

            # Relativistic Average Logic
            real_loss = bce(real_out - torch.mean(fake_out), torch.ones_like(real_out))
            fake_loss = bce(fake_out - torch.mean(real_out), torch.zeros_like(fake_out))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            # ---------------------
            # Train Generator (RaGAN)
            # ---------------------
            optim_G.zero_grad()
            sr_img = G(lr_img) # Re-generate with gradients attached
            fake_out = D(sr_img)
            real_out = D(hr_img).detach() # We don't update D here

            # Pixel & Perceptual losses
            percep_loss = vgg_loss(sr_img, hr_img)
            pix_loss = mae(sr_img, hr_img)

            # Adversarial loss (Symmetrical form for Generator)
            # We want: fake to be "more real" than avg real AND real to be "less real" than avg fake
            g_real_loss = bce(real_out - torch.mean(fake_out), torch.zeros_like(real_out))
            g_fake_loss = bce(fake_out - torch.mean(real_out), torch.ones_like(fake_out))
            adv_loss = (g_real_loss + g_fake_loss) / 2

            # Total Generator Loss
            # Weights recommended by paper: lambda=5e-3 (adv), eta=1e-2 (pixel)
            g_loss = percep_loss + w_adv * adv_loss + w_pix * pix_loss

            g_loss.backward()
            optim_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

            loop.set_postfix({
                "D": f"{d_loss.item():.4f}",
                "G": f"{g_loss.item():.4f}",
            })

            # Log batch metrics
            if use_wandb:
                wandb.log({
                    "train/d_loss": d_loss.item(),
                    "train/d_loss_real": real_loss.item(),
                    "train/d_loss_fake": fake_loss.item(),
                    "train/g_loss": g_loss.item(),
                    "train/adversarial_loss": adv_loss.item(),
                    "train/perceptual_loss": percep_loss.item(),
                    "train/pixel_loss": pix_loss.item(),
                })

        # Log epoch averages
        if use_wandb:
            wandb.log({
                "train/epoch": epoch,
                "train/avg_d_loss": epoch_d_loss / num_batches,
                "train/avg_g_loss": epoch_g_loss / num_batches,
            })

        # Save sample image from last batch
        if save_samples:
            save_sample(sr_img, epoch)
            if use_wandb:
                wandb.log({"train/sample": wandb.Image(sr_img[0].detach().cpu())})

        # ---------------------
        # Validation PSNR
        # ---------------------
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                psnr_vals = []
                for lr_img, hr_img in val_loader:
                    lr_img = lr_img.to(device)
                    hr_img = hr_img.to(device)
                    sr_img = G(lr_img)
                    psnr_vals.append(psnr(sr_img, hr_img))
                mean_psnr = sum(psnr_vals) / len(psnr_vals)
                print(f"Validation PSNR after epoch {epoch}: {mean_psnr:.2f} dB")
                
                if use_wandb:
                    wandb.log({
                        "val/psnr": mean_psnr,
                        "val/epoch": epoch,
                    })
        else:
            print(f"Epoch {epoch}: validation skipped (no pairs).")

    if use_wandb:
        wandb.finish()

