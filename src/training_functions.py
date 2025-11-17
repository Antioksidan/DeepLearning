import torch
import torch.nn as nn
from torchvision import models, utils
import torch.nn.functional as F
import numpy as np
import math
import tqdm
import os

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

def pretrain_generator(G, train_loader, pretrain_epochs=10, lr=1e-4, save_samples=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # pretrain generator (with mse)
    print("Pretraining generator...")

    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    mse = nn.MSELoss()
    mse.to(device)

    for epoch in range(1, pretrain_epochs + 1):
        G.train()
        loop = tqdm.tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{pretrain_epochs}")
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optim_G.zero_grad()
            sr_img = G(lr_img)

            # perceptual = vgg_loss(sr_img, hr_img)
            pixel_loss = mse(sr_img, hr_img)
            pixel_loss.backward()
            optim_G.step()

            loop.set_postfix({
                "G": f"{pixel_loss.item():.4f}",
            })

        # Save sample image from last batch
        if save_samples:
            save_sample(sr_img, epoch, out_dir="pretrain_samples")
    
    return G

def train(G, D, train_loader, val_loader=None, num_epochs=50, lr=1e-4, save_samples=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    vgg_loss = VGGLoss().to(device)
    bce = nn.BCELoss().to(device)
    mse = nn.MSELoss().to(device)

    optim_G = torch.optim.Adam(G.parameters(), lr=lr)
    optim_D = torch.optim.Adam(D.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()

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
            g_loss = perceptual + 0.01 * pixel_loss + 1e-3 * adv_loss

            g_loss.backward()
            optim_G.step()

            loop.set_postfix({
                "D": f"{d_loss.item():.4f}",
                "G": f"{g_loss.item():.4f}",
            })

        # Save sample image from last batch
        if save_samples:
            save_sample(sr_img, epoch)

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
        else:
            print(f"Epoch {epoch}: validation skipped (no pairs).")

def save_models(G, D):
    os.makedirs("models", exist_ok=True)
    torch.save(G.state_dict(), "models/generator_srgan.pth")
    torch.save(D.state_dict(), "models/discriminator_srgan.pth")
    print("Models saved as models/generator_srgan.pth and models/discriminator_srgan.pth")