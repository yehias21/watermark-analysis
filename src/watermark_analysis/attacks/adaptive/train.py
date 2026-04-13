"""Adaptive VAE-attack training loop."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import lpips
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm, trange

from ...config import (
    ADAPTIVE_MAX_SAMPLES_TO_PLOT,
    ADAPTIVE_RUN_SUBDIRS,
    ADAPTIVE_RUNS_DIR,
    ADAPTIVE_TRAIN_VAL_SPLIT,
    ADAPTIVE_WANDB_PROJECT,
    DEFAULT_NUM_WORKERS,
)
from .model import build_loading_transform, load_pretrained_vae


def _load_single_image(image_path, transform):
    img = Image.open(image_path)
    return transform(img)


def load_attack_dataset(path: str, num_workers: int = DEFAULT_NUM_WORKERS):
    transform = build_loading_transform()
    csv_file = os.path.join(path, "messages.csv")
    data = pd.read_csv(csv_file)

    no_watermark_paths = [os.path.join(path, "no_watermark", f"data_{idx}.png") for idx in data["index"]]
    watermark_paths = [os.path.join(path, "watermark", f"data_{idx}.png") for idx in data["index"]]
    inverse_watermark_paths = [os.path.join(path, "inverse_watermark", f"data_{idx}.png") for idx in data["index"]]

    def load_image_set(paths):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(tqdm(
                executor.map(lambda p: _load_single_image(p, transform), paths),
                total=len(paths),
                desc="Loading images",
            ))
        return torch.stack(images)

    print("Loading no watermark images...")
    no_watermark = load_image_set(no_watermark_paths)
    print("Loading watermark images...")
    watermark_images = load_image_set(watermark_paths)
    print("Loading inverse watermark images...")
    inverse_watermark_images = load_image_set(inverse_watermark_paths)

    messages = torch.tensor([eval(m) for m in data["message"]])
    return no_watermark, watermark_images, inverse_watermark_images, messages


def create_run_directory() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ADAPTIVE_RUNS_DIR, f"run_{timestamp}")
    for subdir in tqdm(ADAPTIVE_RUN_SUBDIRS, desc="Creating directories"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    return run_dir


def save_sample_images(x_wm: torch.Tensor, recon_x: torch.Tensor, epoch: int, run_dir: str) -> None:
    samples_dir = os.path.join(run_dir, "samples")
    with tqdm(total=3, desc=f"Saving samples for epoch {epoch}") as pbar:
        x_wm = (x_wm.cpu().detach() + 1.0) / 2.0
        recon_x = (recon_x.cpu().detach() + 1.0) / 2.0
        pbar.update(1)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        n_wm = min(ADAPTIVE_MAX_SAMPLES_TO_PLOT, x_wm.size(0))
        n_recon = min(ADAPTIVE_MAX_SAMPLES_TO_PLOT, recon_x.size(0))
        axes[0].imshow(torch.cat([x_wm[i] for i in range(n_wm)], dim=2).permute(1, 2, 0))
        axes[0].set_title("Watermarked Images")
        axes[0].axis("off")
        axes[1].imshow(torch.cat([recon_x[i] for i in range(n_recon)], dim=2).permute(1, 2, 0))
        axes[1].set_title("Reconstructed Images")
        axes[1].axis("off")
        pbar.update(1)

        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"comparison_epoch_{epoch:03d}.png"))
        plt.close()
        pbar.update(1)


def train_adaptive_attack(
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    alpha: float = 1,
    beta: float = 1,
    cache_dir: str = "cache/attack_dataset_rivagan",
    mode: str = "inverse_watermark",
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    run_dir = create_run_directory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading attack dataset...")
    no_watermark, watermark_images, inverse_watermark_images, _ = load_attack_dataset(cache_dir)
    logger.info("Attack dataset loaded.")
    if mode == "inverse_watermark":
        target_images = inverse_watermark_images
    elif mode == "no_watermark":
        target_images = no_watermark
    else:
        raise ValueError(f"Unknown mode: {mode}")

    watermark_images = watermark_images.float() * 2 - 1.0
    target_images = target_images.float() * 2 - 1.0

    dataset = TensorDataset(watermark_images, target_images)
    train_size = int(ADAPTIVE_TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vae = load_pretrained_vae(device)
    lpips_loss = lpips.LPIPS(net="alex").to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    wandb.init(
        project=ADAPTIVE_WANDB_PROJECT,
        name=f"config_alpha_{alpha}_beta_{beta}_mode_{mode}_dataset_{cache_dir}",
    )
    wandb.config.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "beta": beta,
        "cache_dir": cache_dir,
        "mode": mode,
    })
    wandb.watch(vae)

    for epoch in trange(num_epochs, desc="Training"):
        vae.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False, position=1)
        for wm_batch, target_batch in train_pbar:
            target_batch = target_batch.to(device)
            wm_batch = wm_batch.to(device)
            optimizer.zero_grad()
            latents = vae.encode(wm_batch).latent_dist.sample()
            recon_x = vae.decode(latents).sample
            loss = alpha * lpips_loss(recon_x, target_batch).mean() + beta * mse_loss(recon_x, target_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * wm_batch.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        vae.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False, position=1)
        with torch.no_grad():
            for wm_batch, target_batch in val_pbar:
                wm_batch = wm_batch.to(device)
                target_batch = target_batch.to(device)
                latents = vae.encode(wm_batch).latent_dist.sample()
                recon_x = vae.decode(latents).sample
                loss = alpha * lpips_loss(recon_x, target_batch).mean() + beta * mse_loss(recon_x, target_batch)
                val_loss += loss.item() * wm_batch.size(0)
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            save_sample_images(wm_batch, recon_x, epoch, run_dir)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
        wandb.log({"epoch": epoch + 1})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(vae.state_dict(), os.path.join(run_dir, "models", "best_model.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.savefig(os.path.join(run_dir, "plots", "loss_curves.png"))
        plt.close()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    logger.info(
        f"Training complete. Best model saved at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}"
    )
    logger.info(f"Training outputs saved to: {run_dir}")
    return train_losses, val_losses
