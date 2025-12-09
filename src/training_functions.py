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
from src.generator import Generator
# Discriminator network
from src.discriminator import Discriminator
# module for VGG-based perceptual loss
from src.vgg_wrapper import VGGLoss

# =========================================================
# METRICS
# =========================================================
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10((max_val ** 2) / mse)

# =========================================================
# TRAINING
# =========================================================
def save_sample(sr_batch, epoch, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    # save first image in batch
    img = sr_batch[0].detach().cpu()  # [3,H,W], in [0,1] thanks to Sigmoid
    utils.save_image(img, os.path.join(out_dir, f"sr_epoch_{epoch:03d}.png"))

def pretrain_generator(G, train_loader, pretrain_epochs=10, lr=1e-4, save_samples=False, use_wandb=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize wandb if enabled
    if use_wandb:
        timestamp = int(time.time())
        wandb.init(
            name=f"pretrain_generator_{timestamp}",
            project="veo_srgan", 
            config={
            "pretrain_epochs": pretrain_epochs,
            "device": device,
            "lr": lr,
            "architecture": "Generator",
        })
        wandb.watch(G, log="all", log_freq=100)

    # pretrain generator (with mse)
    print("Pretraining generator...")

    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    mse = nn.MSELoss()
    mse.to(device)

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

            # perceptual = vgg_loss(sr_img, hr_img)
            pixel_loss = mse(sr_img, hr_img)
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

def train(G, D, train_loader, val_loader=None, 
          num_epochs=50, 
          lr=1e-4, 
          w_pix=0.01, w_adv=1e-3, 
          save_samples=False, 
          use_wandb=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize wandb if enabled
    if use_wandb:
        timestamp = int(time.time())
        wandb.init(
            name=f"train_srgan_{timestamp}",
            project="veo_srgan", 
            config={
            "num_epochs": num_epochs,
            "device": device,
            "lr": lr,
            "w_pix": w_pix,
            "w_adv": w_adv,
            "architecture": "SRGAN",
        })
        wandb.watch([G, D], log="all", log_freq=100)

    vgg_loss = VGGLoss().to(device)
    bce = nn.BCELoss().to(device)
    mse = nn.MSELoss().to(device)

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
            # Train Discriminator
            # ---------------------
            optim_D.zero_grad()

            real_out = D(hr_img)
            real_labels = torch.ones_like(real_out)
            d_loss_real = bce(real_out, real_labels)

            sr_img = G(lr_img).detach()
            fake_out = D(sr_img)
            fake_labels = torch.zeros_like(fake_out)
            d_loss_fake = bce(fake_out, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optim_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optim_G.zero_grad()
            sr_img = G(lr_img)

            # adversarial loss - try to fool discriminator
            pred_fake = D(sr_img)
            adv_loss = bce(pred_fake, torch.ones_like(pred_fake))

            # content losses
            perceptual = vgg_loss(sr_img, hr_img)
            pixel_loss = mse(sr_img, hr_img)

            # combine (weights can be tuned)
            g_loss = perceptual + w_pix * pixel_loss + w_adv * adv_loss

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
                    "train/d_loss_real": d_loss_real.item(),
                    "train/d_loss_fake": d_loss_fake.item(),
                    "train/g_loss": g_loss.item(),
                    "train/adversarial_loss": adv_loss.item(),
                    "train/perceptual_loss": perceptual.item(),
                    "train/pixel_loss": pixel_loss.item(),
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

def save_models(G, D, uid=None):
    os.makedirs("models", exist_ok=True)
    if uid is None:
        uid = int(time.time())
    torch.save(G.state_dict(), f"models/generator_{uid}.pth")
    torch.save(D.state_dict(), f"models/discriminator_{uid}.pth")
    print(f"Models saved as models/generator_{uid}.pth and models/discriminator_{uid}.pth")